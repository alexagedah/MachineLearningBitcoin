# Machine-Learning-Bitcoin
On-chain analysis gained popularity during the crypto bullmarket in 2020 - 2021. Does this analysis actually have predictive power?
Can on-chain analysis predict tomorrow's return on bitcoin?

In this project I attempt to use a multiple linear regression model to forecast the next day's log returns of Bitcoin using on-chain data as the features. 

Below are the on-chain metrics used.
- AVBLS : The Average block size in MB
- BLCHS : The total size of all block headers and transactions. Not including database indexes.
- CPTRA : Data showing miners revenue divided by the number of transactions.
- CPTRV : Data showing miners revenue as as percentage of the transaction volume.
- DIFF : Difficulty is a measure of how difficult it is to find a hash below a given target.
- ETRAV : Similar to the total output volume with the addition of an algorithm which attempts to remove change from the total value. This may be a more accurate reflection of the true transaction volume.
- ETRVU : Similar to the total output volume with the addition of an algorithm which attempts to remove change from the total value. This may be a more accurate reflection of the true transaction volume.
- HRATE : The estimated number of giga hashes per second (billions of hashes per second) the bitcoin network is performing.
- MIREV : Historical data showing (number of bitcoins mined per day + transaction fees) * market price.
- MKPRU : Data showing the USD market price from Mt.gox
- MKTCP : Data showing the total number of bitcoins in circulation the market price in USD.
- MWNUS : Number of wallets hosts using our My Wallet Service.
- NADDU : Number of unique bitcoin addresses used per day.
- NTRAN : Total number of unique bitcoin transactions per day.
- NTRAT : Total number of unique bitcoin transactions per day.
- NTRBL : The average number of transactions per block.
- NTREP : Data showing the total number of unique bitcoin transactions per day excluding those which involve any of the top 100 most popular addresses popular addresses.
- TOTBC : Data showing the historical total number of bitcoins which have been mined.
- TOUTV : The total value of all transaction outputs per day. This includes coins which were returned to the sender as change.
- TRFEE : Data showing the total BTC value of transaction fees miners earn per day.
- TRFUS : Data showing the total BTC value of transaction fees miners earn per day in USD.
- TRVOU : Data showing the USD trade volume from the top exchanges.


The first step in the project is reading the data from the CSV files and creating a DataFrame containing the containing the log differences of the on-chain metrics and tomorrow's log return on bitcoin.
We then remove all the predictor variables which are collinear
We then fit the multiple linear regression model.

The R2 statistic for the model is negative so (not financial advice) I wouldn't use this model to trade.

Next steps
- Think about the predictor variables and the response more deeply. Why would there be a linear relationship between the log difference in an on-chain metric today and the log returns of bitcoin tomorrow?
- Can we improve the way in which we select which predictor variables to use? Currently forward selection is used to drop half of the predictor variables.
- Should we choose a different target variable? The returns in x days? The average returns over the next x days?
- Are there outliers? If so, how should they be handled? Does it make sense to ignore outliers? 
- Are the high leverage points? Should they be removed?
