def create_custom_dictionary():
    """
    This function prompts the user to enter key-value pairs to create a dictionary.
    
    Returns:
    dict: A dictionary containing the provided key-value pairs.
    """
    custom_dict = {}
    while True:
        key = input("Enter the key (or type 'done' to finish): ")
        if key.lower() == 'done':
            break
        value = input(f"Enter the value for key '{key}': ")
        custom_dict[key] = value
    return custom_dict

custom_dict = create_custom_dictionary()
print(custom_dict)