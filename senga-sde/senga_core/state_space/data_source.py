
class HistoricalDataSource:
    def __init__(self, data=None):
        self.data = data or []

    def fetch(self, limit=None):
        if limit is None:
            return list(self.data)
        return self.data[:limit]

    def add_record(self, record):
        self.data.append(record)
