from datetime import date


class OneMonthSofrFuture(object):
    def __init__(self, reference_start: date, reference_end: date) -> None:
        self.index = 'SOFR'
        self.currency = 'USD'
        self.reference_start = reference_start
        self.reference_end = reference_end


