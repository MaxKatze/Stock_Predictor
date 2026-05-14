import backtrader.feeds as feed

header_row_real = "Date,Adj Close,Close,High,Low,Open,Volume\n"
header_row_adjusted = "Date,Close,High,Low,Open,Volume\n"

class PandasData(feed.PandasData):
    '''
    The ``dataname`` parameter inherited from ``feed.DataBase`` is the pandas
    DataFrame. Uses the index as datetime.
    '''
    params = (
        ('datetime', None),  # Use index as datetime
        ('open', 'Open'),
        ('high', 'High'),
        ('low', 'Low'),
        ('close', 'Close'),
        ('volume', 'Volume'),
        ('openinterest', None),
    )