class Tag:
    _instance = None  # Stores the singleton instance.

    def __new__(cls):
        if cls._instance is None:
            # Create singleton instance.
            cls._instance = super(Tag, cls).__new__(cls)
            cls._instance.tag_id = 0  # Initialize tag_id.
        return cls._instance

    def get_next_tag(self):
        # Return the current tag_id, then increment.
        current_tag = self.tag_id
        self.tag_id += 1
        return current_tag
