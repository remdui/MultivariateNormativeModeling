from config.config_schema import ConfigSchema

class Properties:

    _instance = None

    @classmethod
    def initialize(cls, config):
        """Initializes the Properties object only once."""
        if cls._instance is None:
            cls._instance = Properties(config)

    @classmethod
    def get_instance(cls):
        """Returns the initialized Properties instance."""
        if cls._instance is None:
            raise ValueError("Properties have not been initialized.")
        return cls._instance

    def __init__(self, config: dict):
        """Initialize the Properties object with the provided property map."""
        self.sections: dict[str, Properties.Section] = {}

        # Initialize sections and class attributes
        self.__init_section_attributes()
        self.__init_section_properties(config)

        # Lock the object to prevent further modification
        self._locked = True

    def __init_section_attributes(self):
        """Initialize all section attributes to None.
        This is required to let the IDE know about the attributes."""
        self.system = None
        self.general = None
        self.meta = None
        self.dataset = None
        self.model = None
        self.train = None
        self.scheduler = None

    def __init_section_properties(self, property_map):
        # Dynamically create sections and assign them as immutable sections
        for section_name in dir(ConfigSchema):
            if not section_name.startswith("__") and isinstance(getattr(ConfigSchema, section_name), type):
                section_schema = getattr(ConfigSchema, section_name)
                section_data = property_map.get(section_name.lower(), {}) # Assuming lowercase in property_map
                section = Properties.Section(section_data, section_schema, section_name)
                setattr(self, section_name.lower(), section)  # Set section dynamically
                self.sections[section_name.lower()] = section

    def __setattr__(self, key, value):
        """Prevent modifications to existing attributes after initialization."""
        if hasattr(self, '_locked') and self._locked:
            raise AttributeError(f"Cannot modify immutable property section '{key}'")
        super().__setattr__(key, value)

    def __repr__(self):
        """String representation of all sections for debugging purposes."""
        return f"Properties: \n{''.join(f'{v}\n' for k, v in self.sections.items())}"


    class Section:
        """Define a section of the configuration."""
        def __init__(self, section_map, section_schema, section_name):
            self.section_map = section_map
            self.section_schema = section_schema
            self.section_name = section_name

            # Lock the section after initialization
            self._locked = True

        def __getattr__(self, key):
            """Check whether the config property exists in the schema.
            Return the value from the section map if it exists, otherwise return the default value from the schema."""
            if key not in self.section_map:
                raise AttributeError(f"'{self.section_name}' section does not contain property '{key}'")

            if hasattr(self.section_schema, key):
                return self.section_map.get(key, getattr(self.section_schema, key))
            raise AttributeError(f"'{self.section_name}' section contains unknown property '{key}'. Register the property in the ConfigSchema or check for typos.")

        def __setattr__(self, key, value):
            """Custom attribute setter to prevent modification of section attributes."""
            # Allow setting the section_map, section_schema, and section_name before locking the section
            if key in {'section_map', 'section_schema', 'section_name'}:
                super().__setattr__(key, value)

            # Prevent modification of attributes after initialization
            elif hasattr(self, '_locked') and self._locked:
                raise AttributeError(f"Cannot modify property '{key}' in immutable property section '{self.section_name}'")

            # If not locked, allow setting the attribute (also allows locked attributes to be set)
            else:
                super().__setattr__(key, value)

        def __dir__(self):
            """ Set the existing config keys as the attributes of the class"""
            return list(self.section_map.keys())

        def __repr__(self):
            """Provide a useful string representation of a config section."""
            attrs = {key: self.__getattr__(key) for key in dir(self) if not key.startswith("__")}
            return f"Section({self.section_name}: {attrs})"
