import pandas as pd
import seaborn as sns
# Set link to download csv
csv = 'https://raw.githubusercontent.com/resbaz/' \
      'r-novice-gapminder-files/master/data/' \
      'gapminder-FiveYearData.csv'
# Load Dataframe from csv that's in link
df = pd.read_csv(csv)
# Print Dataframe for test
print '- Printing Dataframe extracted from link -'
print df
print '- Making a pivot table, aggregation = average -'
dfpv = df.pivot_table(index='continent',
                      columns='year',
                      values='lifeExp',
                      aggfunc='mean')
# Print Pivot Table for test
print '- Printing pivot table for test -'
print dfpv
# Generating Heat Map
print '- Generating heat map -'
sns.heatmap(dfpv).get_figure().savefig('assign3.png')