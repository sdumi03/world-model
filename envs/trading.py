"""
Simulated trading environment.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image


class Env:
    def __init__(self):
        self.path = '/content/drive/MyDrive/MuTrade/data/'

        self.files = [
            'EURUSD=X',
            'EURJPY=X',
            'EURAUD=X',
            'EURCAD=X',
            'EURCHF=X',
            'EURGBP=X',
            'EURSEK=X',
            'EURNZD=X',

            'GBPUSD=X',
            'GBPJPY=X',
            'GBPAUD=X',
            'GBPCAD=X',
            'GBPCHF=X',
            'GBPNZD=X',

            'NZDCAD=X',
            'NZDCHF=X',
            'NZDUSD=X',

            'JPY=X',
            'CAD=X',
            'CHF=X',
            'MXN=X',
            'SEK=X',

            'AUDCAD=X',
            'AUDCHF=X',
            'AUDJPY=X',
            'AUDNZD=X',
            'AUDUSD=X',

            'CADCHF=X',
            'CADJPY=X',

            'CHFJPY=X'
        ]

        # define width of candlestick elements
        self.width_candle = 1.0
        self.width_candle_minmax = .25

        # define colors to use
        self.color_up = 'green'
        self.color_down = 'red'

        self.seq_len = 30
        self.pred_len = 5

        self.action_space = 2


    def render(self):
        print()
        print("Percent moved:", self.percent_moved)
        print("Position:", self.position)
        print("revenue:", self.revenue)
        print("Variation:", self.variation)
        print("Portfolio value:", self.portfolio_value)


    def data_to_figure_to_image(self):
        self.next_state = self.next_state.reset_index(drop=True)

        # define up and down prices
        up = self.next_state[self.next_state.close > self.next_state.open]
        down = self.next_state[self.next_state.close <= self.next_state.open]

        plt.figure()

        #plot up prices
        plt.bar(up.index,
                up.close - up.open,
                self.width_candle,
                bottom = up.open,
                color = self.color_up
        )
        plt.bar(up.index,
                up.high - up.close,
                self.width_candle_minmax,
                bottom = up.close,
                color = self.color_up
        )
        plt.bar(up.index,
                up.low - up.open,
                self.width_candle_minmax,
                bottom = up.open,
                color = self.color_up
        )

        #plot down prices
        plt.bar(down.index,
                down.close - down.open,
                self.width_candle,
                bottom = down.open,
                color = self.color_down
        )
        plt.bar(down.index,
                down.high - down.open,
                self.width_candle_minmax,
                bottom = down.open,
                color = self.color_down
        )
        plt.bar(down.index,
                down.low - down.close,
                self.width_candle_minmax,
                bottom = down.close,
                color = self.color_down
        )

        # display candlestick chart
        plt.axis('off')
        # plt.show()

        # get the figure
        plt.tight_layout()
        fig = plt.gcf()

        # figure to image
        buf = BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB').resize((300, 300))

        self.img = img
        self.img_array = np.array(img)

        plt.close()
        fig = None
        buf = None
        img = None


    def reset(self):
        self.first = True
        self.current_index = self.seq_len
        self.next_state = None

        self.percent_moved = 0
        self.position = 0
        self.revenue = 0
        self.variation = 0
        self.portfolio_value = 0

        self.done = False
        self.reward = 0

        f = np.random.randint(len(self.files), size=1)[0]

        data = pd.read_csv(self.path + self.files[f].replace('=X', ''))
        data = data.drop(columns=['Unnamed: 0', '4', '5'])
        data = data.rename(columns={'0': 'open', '1': 'high', '2': 'low', '3': 'close'})

        while True:
            init = np.random.randint(len(data), size=1)[0]
            if init > self.seq_len and init < len(data) - self.pred_len: break

        data = data[init - self.seq_len : init + self.pred_len + 1]
        data = data.reset_index(drop=True)

        self.history = data

        self.next_state = self.history[0 : self.current_index]
        # self.insert_more_data_into_next_state() # Does not required
        self.data_to_figure_to_image()

        # return self.next_state
        return self.img_array


    def insert_more_data_into_next_state(self) -> None:
        self.next_state.insert(4, 'percent_moved', self.percent_moved, True)
        self.next_state.insert(5, 'position', self.position, True)
        self.next_state.insert(6, 'revenue', self.revenue, True)
        self.next_state.insert(7, 'variation', self.variation, True)
        self.next_state.insert(8, 'portfolio_value', self.portfolio_value, True)

    def execute_day(self) -> None:
        # One step forward
        self.current_index += 1

        # Calculate revenue
        self.percent_moved = ((self.history['close'][self.current_index] * 100) / self.history['close'][self.current_index - 1]) - 100
        self.percent_moved = round(self.percent_moved, 6)

        self.revenue = self.percent_moved * self.position
        self.variation = 1 if self.revenue >= 0 else -1

        self.portfolio_value += self.revenue
        self.portfolio_value = round(self.portfolio_value, 6)

        # Get next_state and transform to image
        self.next_state = self.history[self.current_index - self.seq_len : self.current_index]
        # self.insert_more_data_into_next_state() # Does not required
        self.data_to_figure_to_image()


    def step(self, action):

        if self.first:
            self.first = False

            # open long position
            if action == 0: self.position = 1

            # open short position
            if action == 1: self.position = -1

            self.execute_day()

        else:
            # continue with the operation this day
            if action == 0: self.execute_day()

            # change the operation
            if action == 1:
                self.position *= -1
                self.execute_day()

        if self.current_index >= 35: self.done = True

        # if self.done: self.reward = 1 if self.portfolio_value > 0 else -1
        # self.reward = round(self.revenue, 1)

        self.reward = self.variation

        return self.img_array, self.reward, self.done, False, {}
