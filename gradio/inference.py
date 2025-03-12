import os
import time

import numpy as np
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
    if seed != None:
        seed = int(seed)

    # Set seeds for all libraries
    random.seed(seed)  # Python's random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    if torch.cuda.is_available():  # PyTorch CUDA
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True  # Ensure deterministic CUDA operations
        # torch.backends.cudnn.benchmark = False  # Disable benchmarking for reproducibility


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
    dim_pattern = r'(?i)("["_^<>@][^"]*dim.[^"]*")'  # todo maybe 'dim' and not 'dim.'

    # Regex pattern for finding the next dynamic marking (!p!, !mp!, !mf!, etc.)
    dynamic_pattern = f'(!p!|!pp!|!ppp!|!pppp!|!mp!|!mf!|!f!|!ff!|!fff!|!ffff!|!sfz!|{dim_pattern}|{cresc_pattern})'

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
    cresc_pattern = r'(?i)("["_^<>@][^"]*cresc[^"]*")'

    # Regex pattern for finding the next dynamic marking (!p!, !mp!, !mf!, etc.)
    dynamic_pattern = f'(!p!|!pp!|!ppp!|!pppp!|!mp!|!mf!|!f!|!ff!|!fff!|!ffff!|!sfz!|{dim_pattern}|{cresc_pattern})'

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
    global model
    global curr_model_path
    if seed == -1:  # rand seed
        set_seed(None)
        seed = random.randint(0, 100000000)
    print("Seed =", seed)
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

        metadata_K_changed_flag = False
        metadata_M_changed_flag = False

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
                    'K:') and metadata_K not in [None, "", "None"]:
                predicted_patch = [ord(c) for c in f'K:{metadata_K}']
                metadata_K_changed_flag = True
            if not tunebody_flag and patchilizer.decode([predicted_patch]).startswith(
                    'M:') and metadata_M not in [None, "", "None"]:
                predicted_patch = [ord(c) for c in f'M:{metadata_M}']
                metadata_M_changed_flag = True

            # todo Q

            if not tunebody_flag and patchilizer.decode([predicted_patch]).startswith('[r:'):  # start with [r:0/
                tunebody_flag = True

                add_to_start_metadata = ""
                if not metadata_K_changed_flag and metadata_K not in [None, "", "None"]:
                    add_to_start_metadata += f'K:{metadata_K}\n'
                if not metadata_M_changed_flag and metadata_M not in [None, "", "None"]:
                    add_to_start_metadata += f'M:{metadata_M}\n'

                if num_bars != None and num_bars not in [0, "None", "", None]:
                    predicted_patch = [ord(c) for c in add_to_start_metadata + f'[r:0/{int(num_bars)}]']
                else:
                    r0_patch = torch.tensor([ord(c) for c in '[r:0/']).unsqueeze(0).to(device)
                    temp_input_patches = torch.concat([input_patches, r0_patch], axis=-1)
                    set_seed(seed)
                    predicted_patch = model.generate(temp_input_patches.unsqueeze(0),
                                                     top_k=top_k,
                                                     top_p=top_p,
                                                     temperature=temperature)
                    predicted_patch = [ord(c) for c in add_to_start_metadata + '[r:0/'] + predicted_patch
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


# %Romantic
# %Chopin, Frederic
# %Keyboard
# %end
# %%score { ( 1 4 ) | ( 2 3 ) }
# L:1/8
# Q:1/4=108
# M:2/4
# K:G
# V:1 treble nm="Piano" snm="Pno."
# V:4 treble
# V:2 bass
# V:3 bass
# [r:0/86][V:1]"^Allegro moderato"!p! z4|[V:2]z [D,G,B,]2 [D,G,B,]|[V:3]G,,4|
# [r:1/85][V:2]z [D,G,B,]2 [D,G,B,]|:[V:3]G,,4|:
# [r:2/84][V:1]{/G} (.B.d.e.g)|[V:2]z [D,G,B,]2 [D,G,B,]|[V:3]G,,4|
# [r:3/83][V:1](!>!f2 ed)|[V:2]z [D,^G,B,]2 [D,G,B,]|[V:3]G,,4|
# [r:4/82][V:1]{/d} (.g.e.d.B)|[V:2]z [D,G,B,]2 [D,G,B,]|[V:3]G,,4|
# [r:5/81][V:1](!>!d2 cA)|[V:2]z [_E,G,C]2 [E,G,C]|[V:3]G,,4|
# [r:6/80][V:1]{/A} (.c.B.G.E)|[V:2]z [D,G,B,]2 [D,G,B,]|[V:3]G,,4|
# [r:7/79][V:1](!>!G2 FD)|[V:2]z [D,A,C]2 [D,A,C]|[V:3]G,,4|
# [r:8/78][V:1]{/D} (.F.E.C.A,)|[V:2]z [D,G,]2 [D,G,]|[V:3]G,,4|
# [r:9/77][V:1](!>!^A,2 B,G,)|[V:2]z [D,G,]2 [D,G,]|[V:3]G,,4|
# [r:10/76][V:1]{/G} (.B.d.e.g)|[V:2]z [D,G,B,]2 [D,G,B,]|[V:3]G,,4|
# [r:11/75][V:1](!>!f2 ed)|[V:2]z [D,^G,B,]2 [D,G,B,]|[V:3]G,,4|
# [r:12/74][V:1]{/d} (.g.e.d.B)|[V:2]z [D,G,B,]2 [D,G,B,]|[V:3]G,,4|
# [r:13/73][V:1](!>!d2 ^cA)|[V:2]z [E,G,^C]2 [E,G,C]|[V:3]G,,4|
# [r:14/72][V:1]{/A} (.^c.B.G.E)|[V:2]z [E,G,^C]2 [E,G,C]|[V:3]G,,4|
# [r:15/71][V:1](!>!e2 dB)|[V:2]z [F,A,D]2 [F,A,D]|[V:3]F,,4|
# [r:16/70][V:1]{/B} (.d.^c.A.F)|[V:2]z [G,A,E]2 [G,A,]|[V:3]E,,2 A,,2|
# [r:17/69][V:1](!>!E2 D) z:|[V:2]z ([G,A,-^C] [F,A,]) z:|[V:3](D,,2 D,) x:|
# [r:18/68][V:1]!f!"^Animato" (!>!_e2 dc)|[V:2](!>!_E2 DC)|[V:4]z [FA]2 [FA]|
# [r:19/67][V:1](!>!_A2 GF)|[V:2](!>!_A,2 G,F,)|[V:4]z [C_E]2 [CE]|
# [r:20/66][V:1](!>!_E2 D^C)|[V:2](!>!_E,2 D,^C,)|[V:4]z [F,C]2 [F,C]|
# [r:21/65][V:1](!>!C2 B,A,)|[V:2](!>!C,2 B,,A,,)|[V:4]z [D,F,]2 [D,F,]|
# [r:22/64][V:1]!p! (!>!_E2 D_B,)|[V:2]z [_B,,_E,]2 [B,,E,]|[V:3]G,,,4|
# [r:23/63][V:1](!>!_A2 G_E)|[V:2]z [_B,,_E,]2 [B,,E,]|[V:3]G,,,4|
# [r:24/62][V:1](!>!_c2 _BG)|[V:2]z [_B,,_E,]2 [B,,E,]|[V:3]G,,,4|
# [r:25/61][V:1](!>!_e2 _BG)|[V:2]z [_B,,_E,]2 [B,,E,]|[V:3]G,,,4|
# [r:26/60][V:1]!f! (!>!_g2 =f_e)|[V:2][K:treble] (!>!_G2 =F_E)|[V:3][K:treble] x4|[V:4]z [Ac]2 [Ac]|
# [r:27/59][V:1](!>!_c2 _BA)|[V:2][K:bass] (!>!_C2 _B,A,)|[V:3][K:bass] x4|[V:4]z [_E_G]2 [EG]|
# [r:28/58][V:1](!>!_G2 =FE)|[V:2](!>!_G,2 =F,E,)|[V:4]z [A,_E]2 [A,E]|
# [r:29/57][V:1](!>!_E2 DC)|[V:2](!>!_E,2 D,C,)|[V:4]z [=F,A,]2 [F,A,]|
# [r:30/56][V:1]!p! (!>!_G2 =F_D)|[V:2]z [_D,_G,]2 [D,G,]|[V:3]_B,,,4|
# [r:31/55][V:1](!>!_c2 _B_G)|[V:2]z [_D,_G,]2 [D,G,]|[V:3]_B,,,4|
# [r:32/54][V:1](!>!__e2 _d__B)|[V:2]z [_D,_G,]2 [D,G,]|[V:3]_B,,,4|
# [r:33/53][V:1](!>!__g2 _d__B)|[V:2]z [_D,_G,]2 [D,G,]|[V:3]__B,,,4|
# [r:34/52][V:1]"^poco riten."!pp! (!>!_g2 =f_e)|[V:2]z [_E,_G,_A,]2 [E,G,A,]|[V:3]_A,,,4|
# [r:35/51][V:1](!>!_B2 c_a)|[V:2]z [_E,_G,_A,]2 [E,G,A,]|[V:3]_A,,,4|
# [r:36/50][V:1](!>!_g2 =f_e)|[V:2]z [_E,_G,_A,]2 [E,G,A,]|[V:3]_A,,,4|
# [r:37/49][V:1](!>!B2 c_a)|[V:2]z [_E,_G,_A,]2 [E,G,A,]|[V:3]_A,,,4|
# [r:38/48][V:1]"^a tempo"!p!!>(! (f2 _ed)!>)!|[V:2]z [D,F,C]2 [D,F,C]|[V:3]_A,,,4|
# [r:39/47][V:1]!>(! (_b2 af)!>)!|[V:2]z [D,F,C]2 [D,F,C]|[V:3]_A,,,4|
# [r:40/46][V:1]!>(! (f2 _ed)!>)!|[V:2]z [D,F,C]2 [D,F,C]|[V:3]_A,,,4|
# [r:41/45][V:1]!>(! (_b2 af)!>)!|[V:2]z [D,F,C]2 [D,F,C]|[V:3]_A,,,4|
# [r:42/44][V:1]"_cresc." (_e'2 d'c')|[V:2]z [D,F,C]2 [D,F,C]|[V:3]_A,,,4|
# [r:43/43][V:1](_b2 ag)|[V:2]z [_E,G,^C]2 [E,G,C]|[V:3]A,,,4|
# [r:44/42][V:1]"_dim." (f2 _ed)|[V:2]z [F,A,D]2 [F,A,D]|[V:3]A,,,4|
# [r:45/41][V:1](^c2 =cA)|[V:2]z [G,A,E]2 [G,A,F]|[V:3]A,,,4|
# [r:46/40][V:1]!p!{/G} (.B.d.e.g)|[V:2]z [D,G,B,]2 [D,G,B,]|[V:3]G,,4|
# [r:47/39][V:1](!>!f2 ed)|[V:2]z [D,^G,B,]2 [D,G,B,]|[V:3]G,,4|
# [r:48/38][V:1]{/d} (.g.e.d.B)|[V:2]z [D,G,B,]2 [D,G,B,]|[V:3]G,,4|
# [r:49/37][V:1](!>!d2 cA)|[V:2]z [_E,G,C]2 [E,G,C]|[V:3]G,,4|
# [r:50/36][V:1]{/A} (.c.B.G.E)|[V:2]z [D,G,B,]2 [D,G,B,]|[V:3]G,,4|
# [r:51/35][V:1](!>!G2 FD)|[V:2]z [D,A,C]2 [D,A,C]|[V:3]G,,4|
# [r:52/34][V:1]{/D} (.F.E.C.A,)|[V:2]z [D,G,]2 [D,G,]|[V:3]G,,4|
# [r:53/33][V:1](!>!^A,2 B,G,)|[V:2]z [D,G,]2 [D,G,]|[V:3]G,,4|
# [r:54/32][V:1]!f!"^Animato"{/G} (.g._b.c'._d')|[V:2]z [_D,G,_B,]2 [D,G,B,]|[V:3]G,,4|
# [r:55/31][V:1](!>!_d'2 c'_b)|[V:2]z [_D,_G,_B,]2 [D,G,B,]|[V:3]_G,,4|
# [r:56/30][V:1]{/_b} (._d'.b._a._g)|[V:2]z [_D,_G,_B,]2 [D,G,B,]|[V:3]_G,,4|
# [r:57/29][V:1](!>!=f2 _ec)|[V:2]z [_D,_G,_B,]2 [D,G,B,]|[V:3]_G,,4|
# [r:58/28][V:1]{/c} (._e._d.c._B)|[V:2]z [_D,_G,_B,]2 [D,G,B,]|[V:3]_G,,4|
# [r:59/27][V:1](!>!_B2 AF)|[V:2]z [_D,_G,C]2 [D,G,C]|[V:3]_G,,4|
# [r:60/26][V:1]{/F} (.A._G.=F._E)|[V:2]z [_D,_G,C]2 [D,G,C]|[V:3]_G,,4|
# [r:61/25][V:1](!>!_E2 _D_B,)|[V:2]z [_D,_G,]2 [D,G,]|[V:3]_G,,4|
# [r:62/24][V:1]!f!{/G} (.g._b.c'._d')|[V:2]z [_D,G,_B,]2 [D,G,B,]|[V:3]G,,4|
# [r:63/23][V:1](!>!_d'2 c'_b)|[V:2]z [_D,_G,_B,]2 [D,G,B,]|[V:3]_G,,4|
# [r:64/22][V:1]{/_b} (._d'.b._a._g)|[V:2]z [_D,_G,_B,]2 [D,G,B,]|[V:3]_G,,4|
# [r:65/21][V:1](!>!=f2 _ec)|[V:2]z [_D,_G,_B,]2 [D,G,B,]|[V:3]_G,,4|
# [r:66/20][V:1]{/c} (._e._d.c._B)|[V:2]z [_D,_G,_B,]2 [D,G,B,]|[V:3]_G,,4|
# [r:67/19][V:1](!>!_B2 AF)|[V:2]z [_D,_G,C]2 [D,G,C]|[V:3]_G,,4|
# [r:68/18][V:1]{/F} (.A._G.=F._E)|[V:2]z [_D,_G,C]2 [D,G,C]|[V:3]_G,,4|
# [r:69/17][V:1]"_dim." (!>!_E2 _D_B,)|[V:2]z [_D,_G,]2 [D,G,]|[V:3]_G,,4|
# [r:70/16][V:1](!>!_E2 _D_B,)|[V:2]z [_D,_G,]2 [D,G,]|[V:3]_G,,4|
# [r:71/15][V:1](!>!_E2 _D_B,)|[V:2]z [_D,_G,]2 [D,G,]|[V:3]_G,,4|
# [r:72/14][V:1](!>!_E2 _D_B,)|[V:2]z [_D,_G,]2 [D,G,]|[V:3]_G,,4|
# [r:73/13][V:1]!pp! ([_B,D]4-|[V:2]z [D,G,]2 [D,G,]|[V:3]G,,4|
# [r:74/12][V:1][B,D]2 [G,_B,][B,D])|[V:2]z [D,G,]2 [D,G,]|[V:3]G,,4|
# [r:75/11][V:1](!>![C_E]4-|[V:2]z [D,G,]2 [D,G,]|[V:3]G,,4|
# [r:76/10][V:1][CE]2 [_B,D][A,C])|[V:2]z [D,G,]2 [D,G,]|[V:3]G,,4|
# [r:77/9][V:1][_B,D]4-|[V:2]z [D,G,]2 [D,G,]|[V:3]G,,4|
# [r:78/8][V:1][B,D]2 ([G,_B,][B,D])|[V:2]z [D,G,]2 [D,G,]|[V:3]G,,4|
# [r:79/7][V:1](!>![C_E]4-|[V:2]z [D,G,]2 [D,G,]|[V:3]G,,4|
# [r:80/6][V:1][CE]2 [_B,D][A,C])|[V:2]z [D,G,]2 [D,G,]|[V:3]G,,4|
# [r:81/5][V:1][_B,D] z z2|[V:2][G,,D,G,] z z2|
# [r:82/4][V:1]{/^C} .[_B,D] z z2|[V:2].[G,,D,G,] z z2|
# [r:83/3][V:1]{/^C} .[_B,D] z z2|[V:2].[G,,D,G,] z z2|
# [r:84/2][V:1]{/^c} .[_Bd] z z2|[V:2].[G,DG] z z2|
# [r:85/1][V:1][K:bass]!ff! !^![G,,_B,,D,G,]4-|[V:2]!8vb(! !^![G,,,,_B,,,,D,,,]4-|[V:3]!8vb(! x4|[V:4][K:bass] x4|
# [r:86/0][V:1][G,,B,,D,G,] z !fermata!z2|][V:2][G,,,,B,,,,D,,,]!8vb)! z !fermata!z2|][V:3]x!8vb)! x3|]
#