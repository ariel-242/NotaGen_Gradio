import os
import time
import torch
from utils import *
from config import *
from transformers import GPT2Config, LlamaConfig
from abctoolkit.utils import Exclaim_re, Quote_re, SquareBracket_re, Barline_regexPattern
from abctoolkit.transpose import Note_list, Pitch_sign_list
from abctoolkit.duration import calculate_bartext_duration

Note_list = Note_list + ['z', 'x']
curr_model_path = INFERENCE_WEIGHTS_PATH

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

patchilizer = Patchilizer()


def get_model(model_path):
    patch_config = GPT2Config(num_hidden_layers=PATCH_NUM_LAYERS,
                              max_length=PATCH_LENGTH,
                              max_position_embeddings=PATCH_LENGTH,
                              n_embd=HIDDEN_SIZE,
                              num_attention_heads=HIDDEN_SIZE // 64,
                              vocab_size=1)
    byte_config = GPT2Config(num_hidden_layers=CHAR_NUM_LAYERS,
                             max_length=PATCH_SIZE + 1,
                             max_position_embeddings=PATCH_SIZE + 1,
                             hidden_size=HIDDEN_SIZE,
                             num_attention_heads=HIDDEN_SIZE // 64,
                             vocab_size=128)

    model = NotaGenLMHeadModel(encoder_config=patch_config, decoder_config=byte_config)

    print("Parameter Number: " + str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    checkpoint = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()

    return model


model = get_model(curr_model_path)


# Function to set the seed in PyTorch and other libraries
def set_seed(seed):
    # Ensure the seed is an integer
    seed = int(seed)

    # Set seeds for reproducibility
    random.seed(seed)  # Python's random module
    torch.manual_seed(seed)  # PyTorch
    if torch.cuda.is_available():  # CUDA (if using GPU)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    print(f"Seed set to: {seed} (type: {type(seed)})")


def get_previous_voice(text, index):
    """
    Finds the last [V:X] voice declaration before a given index in the text.

    :param text: The ABC notation text.
    :param index: The index position to check before.
    :return: The last found voice declaration (e.g., "[V:1]") or None if not found.
    """
    voice_pattern = r"\[V:\d+\]"  # Pattern for voice declarations like [V:1], [V:2], etc.

    # Search for all voice declarations before the given index
    matches = list(re.finditer(voice_pattern, text[:index]))

    # Return the last match if found
    return matches[-1].group(0) if matches else None


def add_cresc_dynamic_markers(abc_text):
    # Regex pattern for finding the cresc annotation
    cresc_pattern = r'(?i)("["_^<>@][^"]*cresc[^"]*")'

    # Regex pattern for finding the next dynamic marking (!p!, !mp!, !mf!, etc.)
    dynamic_pattern = r'(!p!|!pp!|!ppp!|!pppp!|!mp!|!mf!|!f!|!ff!|!fff!|!ffff!|!sfz!)'

    # Find all crescendo matches
    cresc_matches = list(re.finditer(cresc_pattern, abc_text))

    # If no crescendo found, return original text
    if not cresc_matches:
        return abc_text

    start_indexes = []
    end_indexes = []

    for match in cresc_matches:
        cresc_end = match.end()  # Adjust for previous insertions
        curr_voice = get_previous_voice(abc_text, cresc_end)  # Voice of the found cresc

        start_indexes.append(cresc_end)

        # Search for the next dynamic marking after cresc
        dynamic_matchs = list(re.finditer(dynamic_pattern, abc_text[cresc_end:]))

        if dynamic_matchs:
            for dynamic_match in dynamic_matchs:
                dyn_start = cresc_end + dynamic_match.start()
                # Add only if the previous voice is the same as before
                if curr_voice != None and get_previous_voice(abc_text, dyn_start) != curr_voice:
                    continue

                end_indexes.append(dyn_start)
                break

    modified_text = abc_text
    offset = 0  # Track text changes to adjust indices
    add_indexes = sorted([(i, '!<(!') for i in start_indexes] + [(i, '!<)!') for i in end_indexes], key=lambda x: x[0])
    for i, str in add_indexes:
        modified_text = modified_text[:i + offset] + str + modified_text[i + offset:]
        offset += len(str)

    return modified_text


def add_dim_dynamic_markers(abc_text):
    # Regex pattern for finding the cresc annotation
    dim_pattern = r'(?i)("["_^<>@][^"]*dim.[^"]*")'  # todo maybe 'dim' and not 'dim.'

    # Regex pattern for finding the next dynamic marking (!p!, !mp!, !mf!, etc.)
    dynamic_pattern = r'(!p!|!pp!|!ppp!|!pppp!|!mp!|!mf!|!f!|!ff!|!fff!|!ffff!|!sfz!)'

    # Find all crescendo matches
    dim_matches = list(re.finditer(dim_pattern, abc_text))

    # If no crescendo found, return original text
    if not dim_matches:
        return abc_text

    start_indexes = []
    end_indexes = []

    for match in dim_matches:
        dim_end = match.end()  # Adjust for previous insertions
        curr_voice = get_previous_voice(abc_text, dim_end)  # Voice of the found dim

        start_indexes.append(dim_end)

        # Search for the next dynamic marking after dim
        dynamic_matchs = list(re.finditer(dynamic_pattern, abc_text[dim_end:]))

        if dynamic_matchs:
            for dynamic_match in dynamic_matchs:
                dyn_start = dim_end + dynamic_match.start()
                # Add only if the previous voice is the same as before
                if curr_voice != None and get_previous_voice(abc_text, dyn_start) != curr_voice:
                    continue

                end_indexes.append(dyn_start)
                break

    modified_text = abc_text
    offset = 0  # Track text changes to adjust indices
    add_indexes = sorted([(i, '!>(!') for i in start_indexes] + [(i, '!>)!') for i in end_indexes], key=lambda x: x[0])
    for i, str in add_indexes:
        modified_text = modified_text[:i + offset] + str + modified_text[i + offset:]
        offset += len(str)

    return modified_text


def rest_unreduce(abc_lines):
    tunebody_index = None
    for i in range(len(abc_lines)):
        if '[V:' in abc_lines[i]:
            tunebody_index = i
            break

    metadata_lines = abc_lines[: tunebody_index]
    tunebody_lines = abc_lines[tunebody_index:]

    part_symbol_list = []
    voice_group_list = []
    for line in metadata_lines:
        if line.startswith('%%score'):
            for round_bracket_match in re.findall(r'\((.*?)\)', line):
                voice_group_list.append(round_bracket_match.split())
            existed_voices = [item for sublist in voice_group_list for item in sublist]
        if line.startswith('V:'):
            symbol = line.split()[0]
            part_symbol_list.append(symbol)
            if symbol[2:] not in existed_voices:
                voice_group_list.append([symbol[2:]])
    z_symbol_list = []  # voices that use z as rest
    x_symbol_list = []  # voices that use x as rest
    for voice_group in voice_group_list:
        z_symbol_list.append('V:' + voice_group[0])
        for j in range(1, len(voice_group)):
            x_symbol_list.append('V:' + voice_group[j])

    part_symbol_list.sort(key=lambda x: int(x[2:]))

    unreduced_tunebody_lines = []

    for i, line in enumerate(tunebody_lines):
        unreduced_line = ''

        line = re.sub(r'^\[r:[^\]]*\]', '', line)

        pattern = r'\[V:(\d+)\](.*?)(?=\[V:|$)'
        matches = re.findall(pattern, line)

        line_bar_dict = {}
        for match in matches:
            key = f'V:{match[0]}'
            value = match[1]
            line_bar_dict[key] = value

        # calculate duration and collect barline
        dur_dict = {}
        for symbol, bartext in line_bar_dict.items():
            right_barline = ''.join(re.split(Barline_regexPattern, bartext)[-2:])
            bartext = bartext[:-len(right_barline)]
            try:
                bar_dur = calculate_bartext_duration(bartext)
            except:
                bar_dur = None
            if bar_dur is not None:
                if bar_dur not in dur_dict.keys():
                    dur_dict[bar_dur] = 1
                else:
                    dur_dict[bar_dur] += 1

        try:
            ref_dur = max(dur_dict, key=dur_dict.get)
        except:
            pass  # use last ref_dur

        if i == 0:
            prefix_left_barline = line.split('[V:')[0]
        else:
            prefix_left_barline = ''

        for symbol in part_symbol_list:
            if symbol in line_bar_dict.keys():
                symbol_bartext = line_bar_dict[symbol]
            else:
                if symbol in z_symbol_list:
                    symbol_bartext = prefix_left_barline + 'z' + str(ref_dur) + right_barline
                elif symbol in x_symbol_list:
                    symbol_bartext = prefix_left_barline + 'x' + str(ref_dur) + right_barline
            unreduced_line += '[' + symbol + ']' + symbol_bartext

        unreduced_tunebody_lines.append(unreduced_line + '\n')

    unreduced_lines = metadata_lines + unreduced_tunebody_lines

    return unreduced_lines


def inference_patch(period, composer, instrumentation, num_bars, metadata_K, metadata_M, model_path, seed, top_k, top_p,
                    temperature):
    set_seed(seed)
    if model == None or model_path != curr_model_path:
        curr_model_path = model_path
        model = get_model(curr_model_path)

    prompt_lines = [
        '%' + period + '\n',
        '%' + composer + '\n',
        '%' + instrumentation + '\n']

    while True:

        failure_flag = False

        bos_patch = [patchilizer.bos_token_id] * (PATCH_SIZE - 1) + [patchilizer.eos_token_id]

        start_time = time.time()

        prompt_patches = patchilizer.patchilize_metadata(prompt_lines)
        byte_list = list(''.join(prompt_lines))
        print(''.join(byte_list), end='')

        prompt_patches = [[ord(c) for c in patch] + [patchilizer.special_token_id] * (PATCH_SIZE - len(patch)) for patch
                          in prompt_patches]
        prompt_patches.insert(0, bos_patch)

        input_patches = torch.tensor(prompt_patches, device=device).reshape(1, -1)

        end_flag = False
        cut_index = None

        tunebody_flag = False

        while True:
            set_seed(seed)
            predicted_patch = model.generate(input_patches.unsqueeze(0),
                                             top_k=top_k,
                                             top_p=top_p,
                                             temperature=temperature)

            # metadata parameters
            if not tunebody_flag and patchilizer.decode([predicted_patch]).startswith(
                    'K:') and metadata_K != None and metadata_K != "":
                predicted_patch = [ord(c) for c in f'K:{metadata_K}']
            if not tunebody_flag and patchilizer.decode([predicted_patch]).startswith(
                    'M:') and metadata_M != None and metadata_M != "":
                predicted_patch = [ord(c) for c in f'M:{metadata_M}']
            # todo Q

            if not tunebody_flag and patchilizer.decode([predicted_patch]).startswith('[r:'):  # start with [r:0/
                tunebody_flag = True
                if num_bars != None and num_bars > 0:
                    predicted_patch = [ord(c) for c in f'[r:0/{int(num_bars)}]']
                else:
                    r0_patch = torch.tensor([ord(c) for c in '[r:0/']).unsqueeze(0).to(device)
                    temp_input_patches = torch.concat([input_patches, r0_patch], axis=-1)
                    set_seed(seed)
                    predicted_patch = model.generate(temp_input_patches.unsqueeze(0),
                                                     top_k=top_k,
                                                     top_p=top_p,
                                                     temperature=temperature)
                    predicted_patch = [ord(c) for c in '[r:0/'] + predicted_patch
            if predicted_patch[0] == patchilizer.bos_token_id and predicted_patch[1] == patchilizer.eos_token_id:
                end_flag = True
                break
            next_patch = patchilizer.decode([predicted_patch])

            for char in next_patch:
                byte_list.append(char)
                print(char, end='')

            patch_end_flag = False
            for j in range(len(predicted_patch)):
                if patch_end_flag:
                    predicted_patch[j] = patchilizer.special_token_id
                if predicted_patch[j] == patchilizer.eos_token_id:
                    patch_end_flag = True

            predicted_patch = torch.tensor([predicted_patch], device=device)  # (1, 16)
            input_patches = torch.cat([input_patches, predicted_patch], dim=1)  # (1, 16 * patch_len)

            if len(byte_list) > 102400:
                failure_flag = True
                break
            if time.time() - start_time > 20 * 60:
                failure_flag = True
                break

            if input_patches.shape[1] >= PATCH_LENGTH * PATCH_SIZE and not end_flag:
                print('Stream generating...')
                abc_code = ''.join(byte_list)
                abc_lines = abc_code.split('\n')

                tunebody_index = None
                for i, line in enumerate(abc_lines):
                    if line.startswith('[r:') or line.startswith('[V:'):
                        tunebody_index = i
                        break
                if tunebody_index is None or tunebody_index == len(abc_lines) - 1:
                    break

                metadata_lines = abc_lines[:tunebody_index]
                tunebody_lines = abc_lines[tunebody_index:]

                metadata_lines = [line + '\n' for line in metadata_lines]
                if not abc_code.endswith('\n'):
                    tunebody_lines = [tunebody_lines[i] + '\n' for i in range(len(tunebody_lines) - 1)] + [
                        tunebody_lines[-1]]
                else:
                    tunebody_lines = [tunebody_lines[i] + '\n' for i in range(len(tunebody_lines))]

                if cut_index is None:
                    cut_index = len(tunebody_lines) // 2

                abc_code_slice = ''.join(metadata_lines + tunebody_lines[-cut_index:])
                input_patches = patchilizer.encode_generate(abc_code_slice)

                input_patches = [item for sublist in input_patches for item in sublist]
                input_patches = torch.tensor([input_patches], device=device)
                input_patches = input_patches.reshape(1, -1)

        if not failure_flag:
            abc_text = ''.join(byte_list)

            # unreduce
            abc_lines = abc_text.split('\n')
            abc_lines = list(filter(None, abc_lines))
            abc_lines = [line + '\n' for line in abc_lines]
            try:
                unreduced_abc_lines = rest_unreduce(abc_lines)
            except:
                failure_flag = True
                pass
            else:
                unreduced_abc_lines = [line for line in unreduced_abc_lines if
                                       not (line.startswith('%') and not line.startswith('%%'))]
                unreduced_abc_lines = ['X:1\n'] + unreduced_abc_lines
                unreduced_abc_text = ''.join(unreduced_abc_lines)
                fixed_cresc_abc_text = add_cresc_dynamic_markers(unreduced_abc_text)
                fixed_dim_abc_text = add_dim_dynamic_markers(fixed_cresc_abc_text)
                return fixed_dim_abc_text


if __name__ == '__main__':
    inference_patch('Classical', 'Beethoven, Ludwig van', 'Keyboard')
