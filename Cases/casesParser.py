import pandas as pd
import os
directory = os.fsencode("MexicoData")

casesPerCity = {}
dates = []
# fileNameString = "MexicoData"
for file in sorted(os.listdir(directory)):
	filename = os.fsdecode(file)

	excel = pd.read_csv("MexicoData/{}".format(filename), 
		usecols=["report_date", "location", "data_field", "value"])

	weeklyConfirmed = excel.loc[excel["data_field"] == "weekly_zika_confirmed"]

	weeklyConfirmed["location"].replace(to_replace=["Mexico-Mexico"],
		value="Mexico-State_Of_Mexico",
		inplace=True)
	weeklyConfirmed["location"].replace(to_replace=["Mexico-State-Of-Mexico"],
		value="Mexico-State_Of_Mexico",
		inplace=True)
	weeklyConfirmed["location"].replace(to_replace=["Mexico-Distrito_Federal"],
		value="Mexico-Mexico_City",
		inplace=True)

	weeklyConfirmed.sort_values(by=["location"],inplace=True)
	date = None
	for index,row in weeklyConfirmed.iterrows():
		if(date==None):
			date = row.report_date
		if row.location not in casesPerCity:
			casesPerCity[row.location] = []
		casesPerCity[row.location].append(row.value)
	dates.append(date)

file= open("MexicoCases2015-2018.csv", "w")
file.write("CITY,")
file.write(",".join(dates))
file.write("\n")

for key in sorted(casesPerCity.keys()):
	file.write("{},".format(key))
	file.write(",".join(map(str, casesPerCity[key])))
	file.write("\n")

file.close()