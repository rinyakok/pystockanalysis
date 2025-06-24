
import yfinance as yf
import logging as logging
import pandas as pd
import numpy as np


class Stock():
    """A class to represent a stock and perform various calculations on its historical data.
    Attributes:
        symbol (str): The stock symbol.
        name (str): The name of the stock.
        period (str): The period for which to fetch historical data.
        ticker (yf.Ticker): The yfinance ticker object for the stock.
        historical_data (pd.DataFrame): The historical data of the stock.
        trendline (tuple): A tuple containing x and y values of the trendline.
    """

    def __init__(self, symbol: str, name: str="", period: str="1y"):
        self.symbol = symbol
        self.period = period
        self.name = name
        self.ticker, self.historical_data, self.trendline = None, None, None
        self.data = pd
        self.logger = logging.getLogger(__name__)
    
        self._fetch_data() #fetch historical data

    # Fetch historical stock data using yfinance
    def _fetch_data(self):
        """Fetch historical data using yfinance"""
        self.logger.error(f"Fetching data for {self.symbol} for period {self.period}")

        # Reset ticker and historical data if they already exist
        self.ticker, self.historical_data = None, None

        self.ticker = yf.Ticker(self.symbol)
        
        if self.ticker is None:
            # Log and raise an error if ticker is not found
            msg = f"Ticker {self.symbol} not found."
            self.logger.error(msg)
            raise ValueError(msg)
        else:
            # If name is not provided, try to get it from the ticker info
            if self.name == "":
                self.name = self.ticker.info.get('longName', self.symbol)

            self.historical_data = self.ticker.history(period=self.period)

            if self.historical_data.empty:
                msg = f"No historical data found for ticker: {self.ticker}"
                self.logger.error(msg)
                raise ValueError(msg)
            
            self.logger.error(f"Data fetched successfully")
    
    # Calculate Moving Averages for a given pandas Series.
    def _calculate_moving_averages(self, name, window = 20):
        """ Calculate Moving Averages for a given pandas Series.
        :parame name: Name of the moving average (e.g., 'Moving Average 1', 'Moving Average 2')
        :param window: Window size for the moving average (default is 20)
        """
        self.logger.debug(f"Calculating Moving Average: {name} for {self.symbol} with window size {window}")
        self.historical_data[name] = self.historical_data['Close'].rolling(window=window).mean()
        return


    # Calculate the MACD (Moving Average Convergence Divergence) for a given pandas Series.    
    def _calculate_macd(self, short_window=12, long_window=26, signal_window=9):
        """
        Calculate the MACD (Moving Average Convergence Divergence) for a given pandas Series.
        :param self: self.historical_data
        :param short_window: Short period for MACD calculation (default is 12)
        :param long_window: Long period for MACD calculation (default is 26)
        :param signal_window: Signal line period (default is 9)
        """
        if self.__is_historical_data_fetched():
            series = self.historical_data['Close']
            exp1 = series.ewm(span=short_window, adjust=False).mean()
            exp2 = series.ewm(span=long_window, adjust=False).mean()
            self.historical_data['MACD'] = exp1 - exp2
            self.historical_data['Signal'] = self.historical_data['MACD'].ewm(span=signal_window, adjust=False).mean()
            self.logger.debug(f"MACD and Signal line calculated for {self.symbol}")
        
        return
    
    # Calculate the Relative Strength Index (RSI) for a given pandas Series.
    def _calculate_rsi(self, period=14):
        """
        Calculate the Relative Strength Index (RSI) for a given pandas Series.
        :param series: pandas Series of self.historical_data
        :param period: Number of periods to use for RSI calculation (default is 14)
        """

        self.logger.debug(f"Calculating RSI for {self.symbol}...")

        if self.__is_historical_data_fetched():
            delta = self.historical_data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            self.historical_data['RSI'] = 100 - (100 / (1 + rs))
            self.logger.debug(f"RSI calculated for {self.symbol}")
        
        return

    # Calculate Bollinger Bands for a given pandas Series.
    def _calculate_bollinger_bands(self, window=20):
        """
        Calculate the Bollinger Bands for a given pandas Series.
        :param series: pandas Series of self.historical_data
        :param window: Number of indexes to use for Bollinger Band calculation (default is 20)
        """

        self.logger.debug(f"Calculating Bollinger Bands for {self.symbol}...")

        if self.__is_historical_data_fetched():
            std_dev = self.historical_data['Close'].rolling(window=window).std()
            self.historical_data['Middle Bollinger Band'] = self.historical_data['Close'].rolling(window=window).mean()
            self.historical_data['Upper Bollinger Band'] = self.historical_data['Middle Bollinger Band'] + (std_dev * 2)
            self.historical_data['Lower Bollinger Band'] = self.historical_data['Middle Bollinger Band'] - (std_dev * 2)
            self.logger.info(f"Bollinger Bands calculated for {self.symbol}")
        
        return
    
    # ================ Function for trendlines calculation ================
    # Fast trendline calculation
    def _calculate_trendline(self, start_index, end_index, prolongation=-1):
        """
        Calculate a fast linear trendline for the 'Close' price between start_index and end_index (inclusive).
        Returns x (index/dates) and y (trendline values).
        :param start_index: Start index for the trendline calculation
        :param end_index: End index for the trendline calculation
        :param prolongation: Number of additional indexes to extend the trendline (default is -1, which means prolong until the end of data)
        """

        # if self.__is_historical_data_fetched():
        #     sub_hist = self.historical_data[start_index:end_index + 1]
        #     # If there is less than 2 data points, return None
        #     if len(sub_hist) < 2:
        #         self.trendline = None
        #     x = np.arange(len(sub_hist))
        #     y = sub_hist['Close'].values
        #     coef = np.polyfit(x, y, 1)
        #     trend_y = coef[0] * x + coef[1]
        #     self.trendline = {'index': list(sub_hist.index), 'Trendline': list(trend_y)}
        #     self.logger.debug(f"Trendline calculated for {self.symbol} from {start_index} to {end_index}")
        #     return 
        # else:
        #     self.trendline = None
        # return 
    
        if self.__is_historical_data_fetched():
            sub_hist = self.historical_data[start_index:end_index + 1]

            # If there is less than 2 data points, return None
            if len(sub_hist) < 2:
                self.trendline = None
            x = np.arange(len(sub_hist))
            y = sub_hist['Close'].values
            #Find the linear trendline coeeficients to the data range between start index and end index
            coef = np.polyfit(x, y, 1)
            slope, intercept = coef[0], coef[1]

            if prolongation == 0:
                # use the same range indexes as for x
                x_prolonged = x
            elif prolongation > 0:
                # extend the range indexes
                sub_hist = self.historical_data[start_index:end_index + prolongation + 1]
                x_prolonged = np.arange(len(sub_hist))               
            else:
                # extend the range indexes
                sub_hist = self.historical_data[start_index:]
                x_prolonged = np.arange(len(sub_hist))

            #Calculate the trendline values for the prolonged range
            trend_y = slope * x_prolonged + intercept 

            self.trendline = {'index': list(sub_hist.index), 'Trendline': list(trend_y)}
            self.logger.debug(f"Trendline calculated for {self.symbol} from {start_index} to {end_index}")
            return 
        else:
            self.trendline = None
        return 
    
    # ================ Getter functions ================

    # Getter for stock historical data
    def get_historical_data(self, columns=None):
        if columns is None:
            return self.historical_data
        return self.historical_data[columns]
    
    # Getter historical data indexes
    def get_indexes(self):
        return self.historical_data.index

    # Getter for number of stock historical data indexes
    def get_number_of_idx(self):
        if self.historical_data is None:
            self.logger.error("Historical data is not fetched yet.")
            return 0
        
        return len(self.historical_data.index)
    
    # Getter for moving averages
    def get_indicator(self, type):
        if type not in self.historical_data:
            self.logger.error(f"Indicator {type} is not calculated yet.")
            return None
        
        if self.__is_historical_data_fetched():
            return self.historical_data[type]

        return None
    
    # Getter to next closest index
    def get_next_closest_index(self,date):

        #take only date from datetime
        date = pd.to_datetime(date).date()

        date_idx = 0

        #Loop over indexes
        for index in range (0,self.get_number_of_idx()):
            #take only date from index's datetime
            idx_date = self.historical_data.index[index].date()

            #If date is greater or equals ti index's date, store current index for next iteration
            if date >= idx_date:
                date_idx = index
            else:
                #We found a date from index's date which is greater then input date -> return the index of previous element
                return date_idx
        
        #If loop is over, but input date equals to last index's date then this last elemet shall be return
        if date == idx_date:
            return index
        
        return None
    
    # ================ Private functions ================

    # Private Chekker function for historical data availability
    def __is_historical_data_fetched(self):
        """ Check if historical data is empty.
        This function checks if the historical data is empty and raises an error if it is.
        :return: True if historical data is empty, False otherwise
        """
        if self.historical_data.empty:
            msg = "Historical data is empty. Please fetch data first."
            self.logger.error(msg)
            raise ValueError(msg)
        return False if self.historical_data.empty else True
    

    #............. Private functions for trendlines calculation .............
    # Find the most extreme point (support or resistance) in the data
    # Example usage:
    # support_point = find_extreme_point('support', hist.tail(30), np.array(y_trend))
    # resistance_point = find_extreme_point('resistance', hist.tail(30), np.array(y_trend))
    def __find_extreme_point(self, type, input_data, trendline):
        """
        Finds the most extreme point (support or resistance) in the input_data based on the trendline.
        For 'support', it finds the lowest point in input_data that is closest to the trendline.
        For 'resistance', it finds the highest point in input_data that is closest to the trendline.
        Returns the index, value, and distance of the extreme point.
        """

        if type == 'support':
            diffs = trendline - input_data['Low'].values
            idx = np.argmax(diffs)
            return input_data.index[idx], input_data['Low'].iloc[idx], diffs[idx]
        elif type == 'resistance':
            diffs = input_data['High'].values - trendline
            idx = np.argmax(diffs)
            return input_data.index[idx], input_data['High'].iloc[idx], diffs[idx]
        else:
            return None
    

    def __get_prolonged_sup_res_trendlines(self, slope, intercept, start_index, end_index, prolongation =-1):
        """ Calculate the trendline for the prolonged range.
        :param slope: Slope of the trendline.
        :param intercept: Intercept of the trendline.
        :param start_index: Start index for the sub-histogram.
        :param end_index: End index for the sub-histogram.
        :param prolongation: Number of additional indexes to extend the trendline (default is -1, which means prolong until the end of data).
        :return: The trendline for the prolonged range.
        """

        if prolongation < 0:
            # prolongation is -1 so calculate trendlines until the end of data
            # create a new sublist from historical data to the range from start_index to the end of data
            sub_hist = self.historical_data[start_index:]
        else:
            # create a sub list from the historical data for the range
            # create a new sublist from historical data to the range from start_index to end_index + prolongation
            sub_hist = self.historical_data[start_index:end_index + prolongation + 1]

        # Calculate the trendline for the prolonged range        
        trendline = slope * np.arange(len(sub_hist)) + intercept

        return trendline

    # Optimize the trendline intersections
    def __optimize_trendline_intersections(self, type, slope, extreme_point_idx, start_index, end_index, precision=0.000001, max_iter=100, prolongation=-1):
        """
        Iteratively adjust the slope of a trendline through the extreme point (by index) to minimize the distance
        to the closest Low (for support) or High (for resistance) value.
        param self: The input data containing 'High', 'Low', or 'Close' values in self.historical_data.
        param type: 'support' or 'resistance'.
        param slope: Initial slope of the trendline.
        param extreme_point_idx: Index of the extreme point in the input_data.
        param start_index: Start index for the sub-histogram.
        param end_index: End index for the sub-histogram.
        param input_data: The input data containing 'High', 'Low', or 'Close' values.
        param precision: Precision for the distance calculation.
        param max_iter: Maximum number of iterations to adjust the slope.
        param prolongation: Number of additional indexes to extend the trendline (default is -1, which means prolong until the end of data).
        return: Tuple of the optimized trendline (numpy array) and the date index of the closest point.
        """
        sub_hist = self.historical_data[start_index:end_index + 1]
        n = len(sub_hist)
        best_slope = slope
        delta_slope = 0.01 * abs(slope) if slope != 0 else 0.01

        # Get the y-value of the extreme point
        if type == 'support':
            ref_value = self.historical_data['Low'][extreme_point_idx]
            values = sub_hist['Low'].values
        elif type == 'resistance':
            ref_value = self.historical_data['High'][extreme_point_idx]
            values = sub_hist['High'].values
        else:
            ref_value = self.historical_data['Close'][extreme_point_idx]
            values = sub_hist['Close'].values

        # Convert extreme_point_idx (timestamp) to integer position in sub_hist
        idx_pos = sub_hist.index.get_loc(extreme_point_idx)

        prev_distance = float('inf')
        direction = 1  # 1 for increasing slope, -1 for decreasing


        for iter_count in range(max_iter):
            intercept = ref_value - best_slope * idx_pos
            trendline = best_slope * np.arange(n) + intercept

            distances = np.abs(trendline - values)
            distances[idx_pos] = np.inf  # Ignore the extreme point itself
            min_distance = np.min(distances)
            closest_idx = np.argmin(distances)

            if min_distance <= precision:
                # Best matching line is found
                date_idx = sub_hist.index[closest_idx]

                if prolongation != 0:
                    # Prolongation is required, recalculate the trendline for the prolonged range
                    trendline = self.__get_prolonged_sup_res_trendlines(best_slope, intercept, start_index, end_index, prolongation = prolongation)

                return trendline, date_idx

            if min_distance > prev_distance:
                direction *= -1
                delta_slope *= 0.5

            if type == 'resistance':
                best_slope += direction * delta_slope
            elif type == 'support':
                best_slope -= direction * delta_slope

            prev_distance = min_distance


        logging.debug(f"Max iteraction reached, no ideal trendline found")
        
        date_idx = sub_hist.index[closest_idx]
        if prolongation != 0:
            # Prolongation is required, recalculate the trendline for the prolonged range
            trendline = self.__get_prolonged_sup_res_trendlines(best_slope, intercept, start_index, end_index, prolongation = prolongation)

        return trendline, date_idx

    # Get optimized support and resistance trendlines
    def _calculate_sup_res_trendlines(self, start_index = 0, end_index = -1, precision=0.000001, prolongation=-1):
        """
        Calculate optimized support and resistance trendlines for the data between start_index and end_index.
        param self: The input data containing 'High', 'Low', or 'Close' values in self.historical_data.
        param start_index: Start index for the sub-histogram.
        param end_index: End index for the sub-histogram.
        param precision: Precision for the distance calculation.
        """

        if self.__is_historical_data_fetched():
            if self.trendline is None:
                self.trendline = self._calculate_trendline(start_index, end_index, prolongation)

            y_trend = self.trendline['Trendline']
            
            y_trend_arr = np.array(self.trendline['Trendline'])

            # We need to find the trendline for the range:
            # At first we need to find the extreme points in the 
            # Create a sub list from the historical elements (most far point from the trendline) within the given range (prolongation is not taken into consideration)
            sub_hist = self.historical_data[start_index:end_index + 1]

            # Find extreme points (pass over sub list of historical data for the range & the trendline sliced list with the same range)
            support_idx, support_val, _ = self.__find_extreme_point('support', sub_hist, y_trend_arr[:len(sub_hist)])
            resistance_idx, resistance_val, _ = self.__find_extreme_point('resistance', sub_hist, y_trend_arr[:len(sub_hist)])

            # Initial slope from fast_trendline
            n = len(sub_hist)
            if n > 1:
                initial_slope = (y_trend[-1] - y_trend[0]) / (n - 1)
            else:
                initial_slope = 0.0

            # Optimize support trendline using the support extreme point index
            self.trendline['Support Trendline'], support_idx_2nd_point = self.__optimize_trendline_intersections(
                'support', initial_slope, support_idx, start_index, end_index, precision=precision, max_iter=500, prolongation=prolongation
            )

            # Optimize resistance trendline using the resistance extreme point index
            self.trendline['Resistance Trendline'], resistance_idx_2nd_point = self.__optimize_trendline_intersections(
                'resistance', initial_slope, resistance_idx, start_index, end_index, precision=precision, max_iter=500, prolongation=prolongation
            )

            return
    #.............. End of private functions for trendlines calculation .............



    def __str__(self):
        return f"Stock(symbol={self.symbol}, name={self.name}, period={self.period})"