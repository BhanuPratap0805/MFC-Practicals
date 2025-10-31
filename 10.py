import numpy as np

def message_to_numbers(message):
    # Map characters to numbers: A=0, B=1,...,Z=25, space=26
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ '
    char_to_num = {c: i for i, c in enumerate(alphabet)}
    message = message.upper()
    nums = [char_to_num.get(c, 26) for c in message]  # default space if not found
    return nums

def numbers_to_message(nums):
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ '
    return ''.join(alphabet[n % 27] for n in nums)

def encode_message(message, enc_matrix):
    nums = message_to_numbers(message)
    n = enc_matrix.shape[0]
    # Pad with spaces to make length a multiple of n
    while len(nums) % n != 0:
        nums.append(26)
    # Group into vectors and encode
    encoded_nums = []
    for i in range(0, len(nums), n):
        vec = np.array(nums[i:i+n])
        coded_vec = enc_matrix.dot(vec)
        encoded_nums.extend(coded_vec)
    return np.array(encoded_nums)

def decode_message(encoded_nums, enc_matrix):
    n = enc_matrix.shape[0]
    dec_matrix = np.linalg.inv(enc_matrix)
    decoded_nums = []
    for i in range(0, len(encoded_nums), n):
        vec = np.array(encoded_nums[i:i+n])
        decoded_vec = dec_matrix.dot(vec)
        decoded_nums.extend(np.round(decoded_vec).astype(int))
    return numbers_to_message(decoded_nums)

# Example usage
message = "Linear algebra is fun"

# Example 3x3 nonsingular matrix
encoding_matrix = np.array([[2, 1, 1],
                            [1, 3, 2],
                            [1, 0, 0]])

encoded = encode_message(message, encoding_matrix)
print("Encoded numeric message:")
print(encoded)

decoded = decode_message(encoded, encoding_matrix)
print("Decoded message:")
print(decoded)
