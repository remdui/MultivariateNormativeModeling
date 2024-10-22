from config.config_schema import ConfigSchema

class Properties():
    """Holds the final configuration (merged from config file and command-line arguments)."""
    def __init__(self, property_map: dict):
        self.sections = {}

        # Initialize sections
        self.__init_section_attributes()

        # Dynamically create sections for all attributes in ConfigSchema
        for section_name in dir(ConfigSchema):
            if not section_name.startswith("__") and isinstance(getattr(ConfigSchema, section_name), type):
                section_schema = getattr(ConfigSchema, section_name)
                section_data = property_map.get(section_name.lower(), {})  # Assuming lowercase in property_map
                section = Properties.Section(section_data, section_schema, section_name)
                setattr(self, section_name.lower(), section)
                self.sections[section_name.lower()] = section

        # Add system section as a top-level attribute

    def __init_section_attributes(self):
        self.system = None
        self.general = None
        self.meta = None
        self.dataset = None
        self.model = None
        self.train = None
        self.scheduler = None

    def __repr__(self):
        """String representation of all sections for debugging purposes."""
        return f"Properties: \n{''.join(f'{v}\n' for k, v in self.sections.items())}"


    class Section:
        """Define a section of the configuration."""
        def __init__(self, section_map, section_schema, section_name):
            self.section_map = section_map
            self.section_schema = section_schema
            self.section_name = section_name

        def __getattr__(self, key):
            """Check whether the config property exists in the schema.
            Return the value from the section map if it exists, otherwise return the default value from the schema."""
            if key not in self.section_map:
                raise AttributeError(f"'{self.section_name}' section does not contain property '{key}'")

            if hasattr(self.section_schema, key):
                return self.section_map.get(key, getattr(self.section_schema, key))
            else:
                raise AttributeError(f"'{self.section_name}' section contains unknown property '{key}'. Register the property in the ConfigSchema or check for typos.")

        def __dir__(self):
            """ Set the existing config keys as the attributes of the class"""
            return list(self.section_map.keys())

        def __repr__(self):
            """Provide a useful string representation of a config section."""
            attrs = {key: self.__getattr__(key) for key in dir(self) if not key.startswith("__")}
            return f"Section({self.section_name}: {attrs})"
