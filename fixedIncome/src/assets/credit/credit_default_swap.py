from datetime import date

class PremiumLeg(object):
    pass

class ProtectionLeg(object):
    pass

class CreditDefaultSwap(object):

    def __init__(self,
                 credit_default_spread: float,
                 principal: int,
                 transaction_date: date,
                 settlement_convention: str,
                 maturity_date: date,
                 business_calendar: str,
                 premium_payment_frequency: str = 'Quarterly'
                 ) -> None:
        self._credit_default_spread = credit_default_spread


    @property
    def credit_default_spread(self):
        return self._credit_default_spread