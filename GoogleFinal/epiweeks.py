import datetime

# Right now it only works for 2017

# An epidemiological week, commonly referred to as an epi week or a CDC week, 
# is simply a standardized method of counting weeks to allow for the comparison of data year after year.
# Definition
# The first epi week of the year ends, by definition, on the first Saturday of January, 
# as long as it falls at least four days into the month. Each epi week begins on a Sunday 
# and ends on a Saturday.

def getFirstSundayEpiWeek(year):

	d = input("Enter first sunday of epidemiological year in DD-MM-YYYY format: ")
	d = [int(i) for i in d.split("-")]
	return datetime.datetime(d[2],d[1],d[0])

def getEpidemiologicalWeeks(year):
	start = getFirstSundayEpiWeek(year)
	nextWeek = getNextWeek(start)
	weeks = [start]
	end = input("Enter last day of epidemiological year in DD-MM-YYYY format: ")
	end = [int(i) for i in end.split("-")]
	end = datetime.datetime(end[2], end[1], end[0])
	while nextWeek <= end:
	# for i in range(51):
		if(nextWeek < datetime.datetime.today()):
			weeks.append(nextWeek)
			nextWeek = getNextWeek(nextWeek)	
	return weeks

def getYMD(timeframe):
	return "{}/{}/{}".format(timeframe.day,timeframe.month,timeframe.year)

def getNextWeek(timeframe):
	return timeframe+datetime.timedelta(days=7)