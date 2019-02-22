import plotly.plotly as py
import cufflinks as cf
import plotly
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

class df_plot(pd.DataFrame):
    
    def __init__(self, df):
        #self.df = df
        #self.columns = df.columns
        #self.index = df.index
        self.nygrej = 'hej'
        
    def plot_plotly(self):
        iplot([{
            'x': self.index,
            'y': self.df[col],
            'name': col
        }  for col in self.columns])

