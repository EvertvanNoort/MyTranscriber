def splitLongFragments(input_rttm_path, output_rttm_path, max_duration):
    """
    Split long fragments in the RTTM file into pieces with a maximum duration.
    
    Args:
        input_rttm_path (str): Path to the input RTTM file.
        output_rttm_path (str): Path to the output RTTM file with split fragments.
        max_duration (float): Maximum duration for each fragment (default is 10 seconds).
    """
    with open(input_rttm_path, 'r') as input_rttm:
        lines = input_rttm.readlines()

    split_lines = []
    new_end = []

    for line in lines:
        parts = line.strip().split()
        duration = float(parts[4])# - float(parts[3])

        # Check if the fragment duration is greater than the maximum duration
        if duration > max_duration:
            # Split the fragment into smaller pieces with a maximum duration
            current_start = float(parts[3])
            current_end = current_start + max_duration

            new_end = duration - max_duration
            new_start = current_start + max_duration

            # Create a new RTTM line for the split fragment
            split_line = f"{parts[0]} {parts[1]} {parts[2]} {current_start:.3f} {max_duration:.3f} {parts[5]} {parts[6]} {parts[7]} {parts[8]} {parts[9]}\n"
            new_line = f"{parts[0]} {parts[1]} {parts[2]} {new_start:.3f} {new_end:.3f} {parts[5]} {parts[6]} {parts[7]} {parts[8]} {parts[9]}\n"

            split_lines.append(split_line)
            split_lines.append(new_line)

        else:
            # The fragment doesn't need splitting, add it as is
            split_lines.append(line)

    # Write the split RTTM lines to the output file
    with open(output_rttm_path, 'w') as output_rttm:
        output_rttm.writelines(split_lines)
    return new_end


def removeShortFragments(input_rttm_path, output_rttm_path, min_duration):
    """
    Remove short fragments like "uhm" or "yes" in the RTTM file.
    
    Args:
        input_rttm_path (str): Path to the input RTTM file.
        output_rttm_path (str): Path to the output RTTM file with split fragments.
        min_duration (float): Minimum duration for each fragment (default is 10 seconds).
    """
    with open(input_rttm_path, 'r') as input_rttm:
        lines = input_rttm.readlines()

    filtered_lines = []
    new_end = []

    for line in lines:
        parts = line.strip().split()
        duration = float(parts[4])

        # Check if the fragment duration is greater than the maximum duration
        if duration > min_duration:
            filtered_lines.append(line)

    with open(output_rttm_path, 'w') as output_rttm:
        output_rttm.writelines(filtered_lines)            


# # Usage example
# input_rttm_path = 'input.rttm'  # Specify the input RTTM file
# output_rttm_path = 'output.rttm'  # Specify the output RTTM file with split fragments
# min_duration = 0.5

# remove_short_fragments(input_rttm_path, output_rttm_path, min_duration)
# max_duration = 10  # Maximum duration for each fragment

# # print(type(new_end))
# new_end = split_long_fragments(input_rttm_path, output_rttm_path, max_duration)

# # run the function as often as necessary
# # this is a weird way to check if it is done
# while type(new_end) != type([]):
#     new_end = split_long_fragments(output_rttm_path, output_rttm_path, max_duration)
#     pass