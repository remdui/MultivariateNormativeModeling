
class Properties:
    """Holds the final configuration (merged from config file and command-line arguments)."""
    def __init__(self, property_map: dict):
        self.property_map = property_map

    def get(self, key: str, default=None):
        """Retrieve a value from the merged configuration map."""
        return self.property_map.get(key, default)

    def set(self, key: str, value):
        """Set or update a value in the configuration map."""
        self.property_map[key] = value

    def display(self):
        """Display the configuration map."""
        if not self.property_map:
            print("Properties Configuration: No configuration available.")
        else:
            print("Properties Configuration:")
            for key, value in self.property_map.items():
                print(f"{key}: {value}")