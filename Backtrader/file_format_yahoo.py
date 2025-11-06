import backtrader.feeds as feed

header_row_real = "Date,Adj Close,Close,High,Low,Open,Volume\n"
header_row_adjusted = "Date,Close,High,Low,Open,Volume\n"

class PandasData(feed.DataBase):
    '''
    The ``dataname`` parameter inherited from ``feed.DataBase`` is the pandas
    DataFrame
    '''
    params = (
        # Possible values for datetime (must always be present)
        #  None : datetime is the "index" in the Pandas Dataframe
        #  -1 : autodetect position or case-wise equal name
        #  >= 0 : numeric index to the colum in the pandas dataframe
        #  string : column name (as index) in the pandas dataframe
        ('datetime', 0),

        # Possible values below:
        #  None : column not present
        #  -1 : autodetect position or case-wise equal name
        #  >= 0 : numeric index to the colum in the pandas dataframe
        #  string : column name (as index) in the pandas dataframe
        ('open', 4),
        ('high', 2),
        ('low', 3),
        ('close', 1),
        ('volume', 5),
        ('openinterest', None),
    )