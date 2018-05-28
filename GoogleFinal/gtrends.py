from pytrends.request import TrendReq
from epiweeks import getEpidemiologicalWeeks,getYMD,getNextWeek
import datetime
# Login to Google. Only need to run this once, the rest of requests will use the same session.
pytrend = TrendReq()

# Create payload and capture API tokens. Only needed for interest_over_time(), interest_by_region() & related_queries()
# pytrend.build_payload(kw_list=['zika'], geo="MX", timeframe="2017-11-1 2017-11-9")

file = open("BR.csv", "w")

weeks = getEpidemiologicalWeeks(2017)
weekEpiArray = []
cities = {}
file.write("CITY,")
file.write(",".join(map(getYMD, weeks))+"\n")
for i in range(len(weeks)):
	week = weeks[i]
	nextWeek = getNextWeek(week)
	year = week.year
	month = week.month
	day = week.day
	nextWeekC = nextWeek - datetime.timedelta(days=1)
	timeframe = "{}-{}-{} {}-{}-{}".format(year,month,day,nextWeekC.year,nextWeekC.month,nextWeekC.day)
	print("{}-{}-{} -> {}-{}-{}".format(year,month,day,nextWeekC.year,nextWeekC.month,nextWeekC.day))
	pytrend.build_payload(kw_list=['zika'], geo="BR", timeframe=timeframe)
	results = pytrend.interest_by_region(resolution="CITY")
	# if a city's search index is 0 it doesn't appear on results, so we have to keep track of it
	tempWeekArray = {}
	for index, row in results.iterrows():
		tempWeekArray[index] = row["zika"]
		if index not in cities:
			cities[index] = index #Store in dictionary to get O(1) access
	weekEpiArray.append(tempWeekArray)
	print("DONE", week)

# Iterate on weekEpiArray and replace non existent cities with zeroes
searchesByCity = {}
for city in cities:
	searchesByCity[city] = []
	for week in weekEpiArray:
		if city not in week:
			searchesByCity[city].append(0)
		else:
			searchesByCity[city].append(week[city])

# Finally write to file
for city in sorted(searchesByCity.keys()):
	file.write("{},".format(city))
	file.write(",".join(map(str,searchesByCity[city])))
	file.write("\n")

file.close()
