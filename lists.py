# import ast
# import re
# #
# # Read the lists from lists.py
# with open('lists.py', 'r') as f:
#     lines = f.readlines()

# master_list = []
# #
# # # Define a regex pattern for a Python list
# list_pattern = re.compile(r'\[.*\]')
# #
# # Process each line
# for line in lines:
#     if "=" in line:
#         current_list_str = line.split('=')[1].strip()
#         # Check if the string matches the list pattern
#         if list_pattern.fullmatch(current_list_str):
#             current_list = ast.literal_eval(current_list_str)
#             master_list.extend(current_list)

# # Append the combined list to the existing lists.py file
# with open('lists.py', 'a') as f:
#     f.write(f'\nmaster_list = {master_list}\n')

# master_list = ['20230502_000000.csv' ]
# master_list = ['csv/' + filename for filename in master_list]
# print(master_list)
