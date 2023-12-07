#######################
##                   ##
##  Nickolaus White  ##
##  Stephen Lee      ##
##  Mark Carravallah ##
##      CSE881       ##
##                   ##
#######################


######################
## Import Libraries ##
######################
from datetime import datetime


#################
##  Functions  ##
#################

# Categorize time labels
def categorize_time(time_str):
    tm = datetime.strptime(time_str, '%H:%M:%S').time()
    if tm.hour < 6:
        return 'Late Night'
    elif 6 <= tm.hour < 12:
        return 'Morning'
    elif 12 <= tm.hour < 18:
        return 'Afternoon'
    elif 18 <= tm.hour:
        return 'Evening'

# Fix the pricing format issues
def fix_price(price):
    return price.replace('$', '').replace(',', '')


