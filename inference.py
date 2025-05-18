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
import difflib
import re

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
    cresc_pattern = r'("["_^<>@][^"]*cresc[^"]*")'
    dim_pattern = r'("["_^<>@][^"]*dim.[^"]*")'  # todo maybe 'dim' and not 'dim.'

    # Regex pattern for finding the next dynamic marking (!p!, !mp!, !mf!, etc.)
    dynamic_pattern = f'(!p!|!pp!|!ppp!|!pppp!|!mp!|!mf!|!f!|!ff!|!fff!|!ffff!|!sfz!|{dim_pattern}|{cresc_pattern})'

    # Find all crescendo matches
    cresc_matches = list(re.finditer(r"(?i)" + cresc_pattern, abc_text))

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
    dim_pattern = r'("["_^<>@][^"]*dim.[^"]*")'  # todo maybe 'dim' and not 'dim.'
    cresc_pattern = r'("["_^<>@][^"]*cresc[^"]*")'

    # Regex pattern for finding the next dynamic marking (!p!, !mp!, !mf!, etc.)
    dynamic_pattern = f'(!p!|!pp!|!ppp!|!pppp!|!mp!|!mf!|!f!|!ff!|!fff!|!ffff!|!sfz!|{dim_pattern}|{cresc_pattern})'

    # Find all crescendo matches
    dim_matches = list(re.finditer(r"(?i)" + dim_pattern, abc_text))

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
        random.seed(None)
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
                    predicted_patch = [ord(c) for c in add_to_start_metadata + f'[r:0/{int(num_bars)-1}]']
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

def postprocess_inst_names(abc_text):
    with open('standard_inst_names.txt', 'r', encoding='utf-8') as f:
        standard_instruments_list = [line.strip() for line in f if line.strip()]

    with open('instrument_mapping.json', 'r', encoding='utf-8') as f:
        instrument_mapping = json.load(f)

    abc_lines = abc_text.split('\n')
    abc_lines = list(filter(None, abc_lines))
    abc_lines = [line + '\n' for line in abc_lines]

    for i, line in enumerate(abc_lines):
        if line.startswith('V:') and 'nm=' in line:
            match = re.search(r'nm="([^"]*)"', line)
            if match:
                inst_name = match.group(1)

                # Check if the instrument name is already standard
                if inst_name in standard_instruments_list:
                    continue

                # Find the most similar key in instrument_mapping
                matching_key = difflib.get_close_matches(inst_name, list(instrument_mapping.keys()), n=1, cutoff=0.6)

                if matching_key:
                    # Replace the instrument name with the standardized version
                    replacement = instrument_mapping[matching_key[0]]
                    new_line = line.replace(f'nm="{inst_name}"', f'nm="{replacement}"')
                    abc_lines[i] = new_line

    # Combine the lines back into a single string
    processed_abc_text = ''.join(abc_lines)
    return processed_abc_text

if __name__ == '__main__':
    # inference_patch('Classical', 'Beethoven, Ludwig van', 'Keyboard')
    # unreduce
    abc_text= """%Romantic
%Chopin, Frederic
%Keyboard
%end
%%score { ( 1 3 ) | ( 2 4 ) }
L:1/8
Q:1/4=88
M:3/4
K:A
V:1 treble nm="Piano"
V:3 treble 
V:2 bass 
V:4 bass 
[r:0/119][V:1]"^Andante con moto"!p! (c|
[r:1/118][V:1]B/A/G/A/ G/F/^E/F/ =E/D/F/D/)|[V:2]!ped! .F,,2 [C,A,]2!ped-up! [D,A,]2|
[r:2/117][V:1](!>!C>A, F,2) (!>!C2-|[V:2]!ped! .F,,2 [C,A,]2!ped-up! [C,A,]2|
[r:3/116][V:1]C/^D/^E/F/!<(! G/A/B/c/ d/^d/e/!<)!^e/|[V:2]!ped! .C,,2 [C,B,]2!ped-up! [C,B,]2|
[r:4/115][V:1]!>(! g/f/c/A/!>)! F2) z (c-|[V:2]!ped! .F,,2 [C,A,]2!ped-up! [C,A,]2|
[r:5/114][V:1]c/B/g/f/ e/d/c/B/ A/G/B/F/)|[V:2]!ped! .D,,2 [D,B,]2!ped-up! [D,B,]2|
[r:6/113][V:1](!>!^E>G C2) (!>!A2-|[V:2]!ped! .C,,2 [C,B,]2!ped-up! [C,B,]2|
[r:7/112][V:1]A/G/e/d/ c/B/A/G/ F/^E/G/D/)|[V:2]!ped! .B,,,2 [B,,G,]2!ped-up! [B,,G,]2|
[r:8/111][V:1](!>!C>F A,2) (!>!F2-|[V:2]!ped! .A,,,2 [A,,F,]2!ped-up! [A,,F,]2|
[r:9/110][V:1]F/^E/G/B/"_cresc." d/^e/g/b/!8va(! d'/^e'/g'/d''/|[V:2]!ped! .G,,,2 [G,D^E]2!ped-up! [G,DE]2|[V:3]x4!8va(! x2|
[r:10/109][V:1]!>(! c''/b'/g'/^e'/!8va)! c'/b/g/^e/!>)! c/B/G/^E/)|[V:2]!ped! .C,,2 [G,C^E]2!ped-up! z2|[V:3]x2!8va)! x4|
[r:11/108][V:1]!p! (=G/F/A/=c/ ^d/f/a/=c'/!8va(! ^d'/f'/a'/d''/|[V:2]!ped! .A,,,2 [A,^DF]2!ped-up! [A,DF]2|[V:3]x4!8va(! x2|
[r:12/107][V:1]!>(! =c''/a'/f'/^d'/!8va)! =c'/a/f/^d/!>)! =c/A/F/^D/)|[V:2]!ped! .B,,,2 [A,^DF]2!ped-up! z2|[V:3]x2!8va)! x4|
[r:13/106][V:1]!f! (D/C/E/=G/ ^A/c/e/=g/!8va(! ^a/c'/e'/^a'/|[V:2]!ped! .C,,2 [=G,^A,E]2!ped-up! [G,A,E]2|[V:3]x4!8va(! x2|
[r:14/105][V:1]!>(! a'/g'/f'/^d'/!8va)! ^b/a/f/^d/!>)! ^B/A/F/^D/)|[V:2]!ped! .^D,,2 [F,A,^B,]2!ped-up! z2|[V:3]x2!8va)! x4|
[r:15/104][V:1]!p! (=D/C/E/=G/ ^A/c/e/=g/!8va(! ^a/c'/e'/^a'/|[V:2]!ped! .E,,2 [=G,^A,E]2!ped-up! [G,A,E]2|[V:3]x4!8va(! x2|
[r:16/103][V:1]!>(! a'/g'/f'/^d'/!8va)! ^b/a/f/^d/!>)! ^B/A/F/^D/)|[V:2]!ped! .F,,2 [F,A,^B,]2!ped-up! z2|[V:3]x2!8va)! x4|
[r:17/102][V:1]!pp! (D/C/F/^E/ A/G/d/c/ f/^e/a/g/|[V:2]!ped! .C,,2 [C,B,]2!ped-up! z2|
[r:18/101][V:1]!8va(! d'/c'/f'/^e'/ a'/g'/d''/c''/!8va)!!8va(! ^e'')!8va)! z|[V:2]z2 z2[K:treble] z!ped-up! (c|[V:3]!8va(! x4!8va)!!8va(! x!8va)! x|[V:4]x4[K:treble] x2|
[r:19/100][V:1]!pp! z2 [CAc]2 [DAd]2|[V:2]B/A/G/A/ G/F/^E/F/ =E/D/F/D/)|
[r:20/99][V:1]z2 [CAc]2 [CAc]2|[V:2](!>!C>A, F,2) (!>!C2-|
[r:21/98][V:1]z2 [^EBc^e]2 [EBce]2|[V:2]C/^D/^E/F/!<(! G/A/B/c/ d/^d/e/!<)!^e/|
[r:22/97][V:1]z2 [FAcf]2 [FAcf]2|[V:2]!>(! g/f/c/A/!>)! F2) z (c-|
[r:23/96][V:1]z2 [FBf]2 [FBf]2|[V:2]c/B/g/f/ e/d/c/B/ A/G/B/F/)|
[r:24/95][V:1]z2 [^EBc^e]2 [EBce]2|[V:2](!>!^E>G C2) (!>!A2-|
[r:25/94][V:1]z2 [DGd]2 [DGd]2|[V:2]A/G/e/d/ c/B/A/G/ F/^E/G/D/)|
[r:26/93][V:1]z2 [CFc]2 [CFc]2|[V:2](!>!C>F A,2) (!>!F2-|
[r:27/92][V:1]z2!<(! [^EBd^e]2!<)! [debd']2|[V:2]F/^E/G/B/ d/^e/g/b/!8va(! d'/^e'/g'/d''/|[V:4]x4!8va(! x2|
[r:28/91][V:1]!>(! z2 [c^ebc']2!>)! [cebc']2|[V:2]c''/b'/g'/^e'/!8va)! c'/b/g/^e/ c/B/G/^E/)!ped-up!|[V:4]x2!8va)! x4|
[r:29/90][V:1]z2!<(! [^da=c'^d']2!<)! [dac'd']2|[V:2](=G/F/A/=c/ ^d/f/a/=c'/!8va(! ^d'/f'/a'/d''/|[V:4]x4!8va(! x2|
[r:30/89][V:1]!>(! z2 [^da=c'^d']2!>)! [dac'd']2|[V:2]=c''/a'/f'/^d'/!8va)! =c'/a/f/^d/ =c/A/F/^D/)|[V:4]x2!8va)! x4|
[r:31/88][V:1]z2!<(! [e^ae']2!<)! [eae']2|[V:2](D/C/E/=G/ ^A/c/e/=g/!8va(! ^a/c'/e'/^a'/|[V:4]x4!8va(! x2|
[r:32/87][V:1]!>(! z2 [fa^bf']2!>)! [fabf']2|[V:2]a'/g'/f'/^d'/!8va)! ^b/a/f/^d/ ^B/A/F/^D/)|[V:4]x2!8va)! x4|
[r:33/86][V:1]z2!<(! [=g^ae'=g']2!<)! [gae'g']2|[V:2](D/C/E/=G/ ^A/c/e/=g/!8va(! ^a/c'/e'/^a'/|[V:4]x4!8va(! x2|
[r:34/85][V:1]!>(! z2 [gb^e'g']2!>)! [gbe'g']2|[V:2]a'/g'/f'/^d'/!8va)! b/g/f/^d/ B/G/F/^D/)|[V:4]x2!8va)! x4|
[r:35/84][V:1]z2!pp! ([gb^e'g']2 [gbe'g']2|[V:2](D/C/F/^E/ A/G/d/c/ f/^e/a/g/|
[r:36/83][V:1][gb^e'g']2 [gbe'g']2 [gbe'g']2)|[V:2]d'/c'/f'/^e'/!8va(! a'/g'/d''/c''/!8va)!!8va(! ^e'')!8va)! z|[V:4]x2!8va(! x2!8va)!!8va(! x!8va)! x|
[r:37/82][V:1][K:F#]!pp! (a'3 g'f'e'|[V:2][K:F#][K:bass]!ped! z2 [C,A,F]2 [C,A,F]2!ped-up!|[V:3][K:F#] x6|[V:4][K:F#][K:bass] F,,6|
[r:38/81][V:1]{/e'} d'z/c'/ b2 a2)|[V:2]!ped! z2 [C,F,C]2 [C,F,C]2!ped-up!|[V:4]A,,6|
[r:39/80][V:1]{/a} (c'z/b/ d2 e2)|[V:2]!ped! z2 [C,G,C]2 [C,G,C]2!ped-up!|[V:4]E,,6|
[r:40/79][V:1]{/e} (gz/f/ ^B2 c2)|[V:2]!ped! z2 [C,F,C]2 [C,F,C]2!ped-up!|[V:4]A,,6|
[r:41/78][V:1]!8va(! (a'3 g'f'e'|[V:2]!ped! z2 [C,A,F]2 [C,A,F]2!ped-up!|[V:3]!8va(! x6|[V:4]F,,6|
[r:42/77][V:1]{/e'} d'z/c'/ b2 a2)|[V:2]!ped! z2 [C,F,C]2 [C,F,C]2!ped-up!|[V:4]A,,6|
[r:43/76][V:1]{/a} (c'z/^b/ d2 e2)|[V:2]!ped! z2 [C,G,C]2 [C,G,C]2!ped-up!|[V:4]E,,6|
[r:44/75][V:1]{/e} (gz/f/ ^B2 c2)!8va)!|[V:2]!ped! z2 [C,G,C]2 [C,G,C]2!ped-up!|[V:3]x6!8va)!|[V:4]=E,,6|
[r:45/74][V:1]"_cresc." (=e'3 d'c'b|[V:2]!ped! z2 [C,^^F,C]2 [C,F,C]2!ped-up!|[V:4]D,,6|
[r:46/73][V:1]{/b} az/g/ ^^f2 =e2)|[V:2]!ped! z2 [C,G,C]2 [C,G,C]2!ped-up!|[V:4]=E,,6|
[r:47/72][V:1]{/=e} (gz/^^f/ A2 B2)|[V:2]!ped! z2 [D,G,D]2 [D,G,D]2!ped-up!|[V:4]D,,6|
[r:48/71][V:1]{/B} (dz/c/ ^^F2 G2)|[V:2]!ped! z2 [=E,G,=E]2 [E,G,E]2!ped-up!|[V:4]C,,6|
[r:49/70][V:1]!f!!8va(! (b'3 a'g'f'|[V:2]!ped! z2 [D,G,D]2 [D,G,D]2!ped-up!|[V:3]!8va(! x6|[V:4]B,,,6|
[r:50/69][V:1]{/f'} e'z/d'/ c'2 b2)|[V:2]!ped! z2 [E,CE]2 [E,CE]2!ped-up!|[V:4]C,,6|
[r:51/68][V:1]{/b} (d'z/c'/ e2 f2)!8va)!|[V:2]!ped! z2 [F,CF]2 [F,CF]2!ped-up!|[V:3]x6!8va)!|[V:4]A,,6|
[r:52/67][V:1]{/f} (az/g/ ^B2 c2)|[V:2]!ped! z2 [G,CG]2 [G,CG]2!ped-up!|[V:4]E,,6|
[r:53/66][V:1]!f!!8va(! (=e''3 d''c''b'|[V:2]!ped! z2 [^^F,C^^F]2 [F,CF]2!ped-up!|[V:3]!8va(! x6|[V:4]D,,6|
[r:54/65][V:1]{/b'} a'z/g'/ ^^f'2 =e'2)|[V:2]!ped! z2 [G,CG]2 [G,CG]2!ped-up!|[V:4]=E,,6|
[r:55/64][V:1]{/=e'} (g'z/^^f'/ a2 b2)|[V:2]!ped! z2 [G,DG]2 [G,DG]2!ped-up!|[V:4]D,,6|
[r:56/63][V:1]{/b} (d'z/c'/ ^^f2 g2)|[V:2]!ped! z2 [G,CG]2 [G,CG]2!ped-up!|[V:4]=E,,6|
[r:57/62][V:1]{/g} (bz/a/ ^^c2 d2)!8va)!|[V:2]!ped! z2 [G,^B,F]2 [G,B,F]2!ped-up!|[V:3]x6!8va)!|[V:4]D,,6|
[r:58/61][V:1]{/d} (fz/e/ ^^F2 G2)|[V:2]!ped! z2 [G,C]2 [G,C]2!ped-up!|[V:4]C,,6|
[r:59/60][V:1]!pp!{/B} (d3 cBA|[V:2]!ped! z2 [C,G,C]2 [C,G,C]2!ped-up!|[V:3]z2 [EG]2 [EG]2|[V:4]E,,6|
[r:60/59][V:1]{/A} Gz/^^F/ G2 c2)|[V:2]!ped! z2 [C,G,C]2 [C,G,C]2!ped-up!|[V:3]z2 E2 E2|[V:4]E,,6|
[r:61/58][V:1]{/B} (d3 cBA|[V:2]!ped! z2 [C,G,C]2 [C,G,C]2!ped-up!|[V:3]z2 [EG]2 [EG]2|[V:4]E,,6|
[r:62/57][V:1]{/A} Gz/^^F/ G2 c2)|[V:2]!ped! z2 [C,G,C]2 [C,G,C]2!ped-up!|[V:3]z2 E2 E2|[V:4]E,,6|
[r:63/56][V:1]"_cresc."{/B} (=d3 cB=A|[V:2]!ped! z2 [C,G,C]2 [C,G,C]2!ped-up!|[V:3]z2 [EG]2 [EG]2|[V:4]E,,6|
[r:64/55][V:1]{/=A} Gz/^^F/ G2 c2)|[V:2]!ped! z2 [C,G,C]2 [C,G,C]2!ped-up!|[V:3]z2 E2 E2|[V:4]E,,6|
[r:65/54][V:1]{/B} (=d3 cB=A|[V:2]!ped! z2 [C,G,C]2 [C,G,C]2!ped-up!|[V:3]z2 [EG]2 [EG]2|[V:4]E,,6|
[r:66/53][V:1]{/=A} G)z/^^F/ G2 G2||[V:2]!ped! z2 [C,G,C]2 [C,G,C]2!ped-up!||[V:3]z2 E2 E2||[V:4]E,,6||
[r:67/52][V:1][K:A]!f! (c'3 bag|[V:2][K:A]!ped! z2 [C,A,F]2 [C,A,F]2!ped-up!|[V:3][K:A] x6|[V:4][K:A] F,,6|
[r:68/51][V:1]{/g} fz/e/ d2 c2)|[V:2]!ped! z2 [C,F,C]2 [C,F,C]2!ped-up!|[V:4]A,,6|
[r:69/50][V:1]{/c} (ez/d/ F2 G2)|[V:2]!ped! z2 [C,G,C]2 [C,G,C]2!ped-up!|[V:4]^E,,6|
[r:70/49][V:1]{/G} (Bz/A/ ^E2 F2)|[V:2]!ped! z2 [C,F,C]2 [C,F,C]2!ped-up!|[V:4]F,,6|
[r:71/48][V:1]!8va(! (c''3 b'a'g'|[V:2]!ped! z2 [C,A,F]2 [C,A,F]2!ped-up!|[V:3]!8va(! x6|[V:4]F,,6|
[r:72/47][V:1]{/g'} f'z/e'/ d'2 c'2)|[V:2]!ped! z2 [C,F,C]2 [C,F,C]2!ped-up!|[V:4]A,,6|
[r:73/46][V:1]{/c'} (e'z/d'/ f2 g2)|[V:2]!ped! z2 [C,G,C]2 [C,G,C]2!ped-up!|[V:4]^E,,6|
[r:74/45][V:1]{/g} (bz/a/ ^e2 f2)!8va)!|[V:2]!ped! z2 [C,F,C]2 [C,F,C]2!ped-up!|[V:3]x6!8va)!|[V:4]F,,6|
[r:75/44][V:1]!pp!{/=G} (B3 AGF|[V:2]!ped! z2 [A,,E,A,]2 [A,,E,A,]2!ped-up!|[V:3]z2 [CE]2 [CE]2|[V:4]C,,6|
[r:76/43][V:1]{/F} Ez/^D/ E2 A2)|[V:2]!ped! z2 [A,,E,A,]2 [A,,E,A,]2!ped-up!|[V:3]z2 C2 C2|[V:4]C,,6|
[r:77/42][V:1]{/=G} (B3 AGF|[V:2]!ped! z2 [A,,E,A,]2 [A,,E,A,]2!ped-up!|[V:3]z2 [CE]2 [CE]2|[V:4]C,,6|
[r:78/41][V:1]{/F} Ez/^D/ E2 A2)|[V:2]!ped! z2 [A,,E,A,]2 [A,,E,A,]2!ped-up!|[V:3]z2 C2 C2|[V:4]C,,6|
[r:79/40][V:1]"_cresc."{/=G} (_B3 AG=F|[V:2]!ped! z2 [A,,E,A,]2 [A,,E,A,]2!ped-up!|[V:3]z2 [CE]2 [CE]2|[V:4]C,,6|
[r:80/39][V:1]{/=F} Ez/^D/ E2 A2)|[V:2]!ped! z2 [A,,E,A,]2 [A,,E,A,]2!ped-up!|[V:3]z2 C2 C2|[V:4]C,,6|
[r:81/38][V:1]{/=G} (_B3 AG=F|[V:2]!ped! z2 [A,,E,A,]2 [A,,E,A,]2!ped-up!|[V:3]z2 [CE]2 [CE]2|[V:4]C,,6|
[r:82/37][V:1]{/=F} E)z/^D/ E2 A2||[V:2]!ped! z2 [A,,E,A,]2 [A,,E,A,]2!ped-up!||[V:3]z2 C2 C2||[V:4]C,,6||
[r:83/36][V:1][K:F#]!ff! (a3 gfe|[V:2][K:F#]!ped! z2 [C,F,C]2 [C,F,C]2!ped-up!|[V:3][K:F#] x6|[V:4][K:F#] A,,,6|
[r:84/35][V:1]{/e} dz/c/ B2 A2)|[V:2]!ped! z2 [C,F,C]2 [C,F,C]2!ped-up!|[V:4]C,,6|
[r:85/34][V:1]{/A} (cz/B/ D2 E2)|[V:2]!ped! z2 [C,G,C]2 [C,G,C]2!ped-up!|[V:4]E,,6|
[r:86/33][V:1]{/E} (Gz/F/ ^B,2 C2)|[V:2]!ped! z2 [C,G,C]2 [C,G,C]2!ped-up!|[V:4]A,,,6|
[r:87/32][V:1]!8va(! (a'3 g'f'e'|[V:2]!ped! z2 [C,F,C]2 [C,F,C]2!ped-up!|[V:3]!8va(! x6|[V:4]A,,,6|
[r:88/31][V:1]{/e'} d'z/c'/ b2 a2)|[V:2]!ped! z2 [C,F,C]2 [C,F,C]2!ped-up!|[V:4]C,,6|
[r:89/30][V:1]{/a} (c'z/^b/ d2 e2)|[V:2]!ped! z2 [C,G,C]2 [C,G,C]2!ped-up!|[V:4]E,,6|
[r:90/29][V:1]{/e} (gz/f/ ^B2 c2)!8va)!|[V:2]!ped! z2 [C,G,C]2 [C,G,C]2!ped-up!|[V:3]x6!8va)!|[V:4]=E,,6|
[r:91/28][V:1]"_cresc." (=e'3 d'c'b|[V:2]!ped! z2 [C,^^F,C]2 [C,F,C]2!ped-up!|[V:4]D,,6|
[r:92/27][V:1]{/b} az/g/ ^^f2 =e2)|[V:2]!ped! z2 [C,G,C]2 [C,G,C]2!ped-up!|[V:4]=E,,6|
[r:93/26][V:1]{/=e} (gz/^^f/ A2 B2)|[V:2]!ped! z2 [D,G,D]2 [D,G,D]2!ped-up!|[V:4]D,,6|
[r:94/25][V:1]{/B} (dz/c/ ^^F2 G2)|[V:2]!ped! z2 [=E,G,=E]2 [E,G,E]2!ped-up!|[V:4]C,,6|
[r:95/24][V:1]!f!!8va(! (b'3 a'g'f'|[V:2]!ped! z2 [D,G,D]2 [D,G,D]2!ped-up!|[V:3]!8va(! x6|[V:4]B,,,6|
[r:96/23][V:1]{/f'} e'z/d'/ c'2 b2)|[V:2]!ped! z2 [E,CE]2 [E,CE]2!ped-up!|[V:4]C,,6|
[r:97/22][V:1]{/b} (d'z/c'/ e2 f2)!8va)!|[V:2]!ped! z2 [F,CF]2 [F,CF]2!ped-up!|[V:3]x6!8va)!|[V:4]A,,6|
[r:98/21][V:1]{/f} (az/g/ ^B2 c2)|[V:2]!ped! z2 [G,CG]2 [G,CG]2!ped-up!|[V:4]E,,6|
[r:99/20][V:1]!f!!8va(! (=e''3 d''c''b'|[V:2]!ped! z2 [^^F,C^^F]2 [F,CF]2!ped-up!|[V:3]!8va(! x6|[V:4]D,,6|
[r:100/19][V:1]{/b'} a'z/g'/ ^^f'2 =e'2)|[V:2]!ped! z2 [G,CG]2 [G,CG]2!ped-up!|[V:4]=E,,6|
[r:101/18][V:1]{/=e'} (g'z/^^f'/ a2 b2)|[V:2]!ped! z2 [G,DG]2 [G,DG]2!ped-up!|[V:4]D,,6|
[r:102/17][V:1]{/b} (d'z/c'/ ^^f2 g2)|[V:2]!ped! z2 [G,CG]2 [G,CG]2!ped-up!|[V:4]=E,,6|
[r:103/16][V:1]{/g} (bz/a/ ^^c2 d2)!8va)!|[V:2]!ped! z2 [G,^B,F]2 [G,B,F]2!ped-up!|[V:3]x6!8va)!|[V:4]D,,6|
[r:104/15][V:1]{/d} (fz/e/ ^^F2 G2)|[V:2]!ped! z2 [G,C]2 [G,C]2!ped-up!|[V:4]C,,6|
[r:105/14][V:1]!pp!{/B} (d3 cBA|[V:2]!ped! z2 [C,G,C]2 [C,G,C]2!ped-up!|[V:3]z2 [EG]2 [EG]2|[V:4]E,,6|
[r:106/13][V:1]{/A} Gz/^^F/ G2 c2)|[V:2]!ped! z2 [C,G,C]2 [C,G,C]2!ped-up!|[V:3]z2 E2 E2|[V:4]E,,6|
[r:107/12][V:1]{/B} (d3 cBA|[V:2]!ped! z2 [C,G,C]2 [C,G,C]2!ped-up!|[V:3]z2 [EG]2 [EG]2|[V:4]E,,6|
[r:108/11][V:1]{/A} Gz/^^F/ G2 c2)|[V:2]!ped! z2 [C,G,C]2 [C,G,C]2!ped-up!|[V:3]z2 E2 E2|[V:4]E,,6|
[r:109/10][V:1]"_cresc."{/B} (=d3 cB=A|[V:2]!ped! z2 [C,G,C]2 [C,G,C]2!ped-up!|[V:3]z2 [EG]2 [EG]2|[V:4]E,,6|
[r:110/9][V:1]{/=A} Gz/^^F/ G2 c2)|[V:2]!ped! z2 [C,G,C]2 [C,G,C]2!ped-up!|[V:3]z2 E2 E2|[V:4]E,,6|
[r:111/8][V:1]{/B} (=d3 cB=A|[V:2]!ped! z2 [C,G,C]2 [C,G,C]2!ped-up!|[V:3]z2 [EG]2 [EG]2|[V:4]E,,6|
[r:112/7][V:1]{/=A} G)z/^^F/ G2 G2||[V:2]!ped! z2 [C,G,C]2 [C,G,C]2!ped-up!||[V:3]z2 E2 E2||[V:4]E,,6||
[r:113/6][V:1][K:A]!ff!!8va(! (c''3 b'a'g'|[V:2][K:A]!ped! z2 [C,A,F]2 [C,A,F]2!ped-up!|[V:3][K:A]!8va(! x6|[V:4][K:A] F,,6|
[r:114/5][V:1]{/g'} f'z/e'/ d'2 c'2)|[V:2]!ped! z2 [C,F,C]2 [C,F,C]2!ped-up!|[V:4]A,,6|
[r:115/4][V:1]{/c'} (e'z/d'/ f2 g2)|[V:2]!ped! z2 [C,G,C]2 [C,G,C]2!ped-up!|[V:4]^E,,6|
[r:116/3][V:1]{/g} (bz/a/ ^e2 f2)!8va)!|[V:2]!ped! z2 [C,F,C]2 [C,F,C]2!ped-up!|[V:3]x6!8va)!|[V:4]F,,6|
[r:117/2][V:1]{/f} (az/g/ c2 d2)|[V:2]!ped! z2 [C,F,C]2 [C,F,C]2!ped-up!|[V:4]F,,6|
[r:118/1][V:1]{/d} (fz/e/ ^A2 B2)|[V:2]!ped! z2 [C,G,C]2 [C,G,C]2!ped-up!|[V:4]F,,6|
[r:119/0][V:1][Q:1/4=48]"^riten."{/B} (dz/c/ ^E2 !fermata!F2)|][V:2]!ped! z2 [C,A,C]4!ped-up!|][V:4]F,,6|]"""
    abc_lines = abc_text.split('\n')
    abc_lines = list(filter(None, abc_lines))
    abc_lines = [line + '\n' for line in abc_lines]
    unreduced_abc_lines = rest_unreduce(abc_lines)

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