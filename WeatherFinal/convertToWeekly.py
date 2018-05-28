filename = input("Enter filename: ")

header = None
startDate = input("Enter startDate in [YYYY-MM-DD] format: ")
endDate = input("Enter endDate in [YYYY-MM-DD] format: ")
coordinates = None

dictData = {"precipProbability": [],
			"precipIntensity": [],
			"precipIntensityMax": [],
			"temperatureHigh": [],
			"temperatureLow": [],
			"humidity": []}

outputFilename = "./Weekly-{}.csv".format(filename.split(".csv")[0])
output = open(outputFilename, "w")
output.write("coordinates,date,precipProbability,precipIntensity,precipIntensityMax,temperatureHigh,temperatureLow,humidity\n")

startParse = False
day = 0

def writeToFile(file, data, coordinates, date):

	file.write("{},".format(coordinates))
	file.write("{},".format(date))

	for key in data:
		data[key] = list(filter(("NA").__ne__, data[key]))
		data[key] = list(filter(("NA\n").__ne__, data[key]))

	if(len(data["precipProbability"]) == 0):
		file.write("{},".format("NA"))
	else:
		file.write("{},".format(sum(data["precipProbability"])/len(data["precipProbability"])))

	if(len(data["precipIntensity"]) == 0):
		file.write("{},".format("NA"))
	else:
		file.write("{},".format(sum(data["precipIntensity"])/len(data["precipIntensity"])))

	if(len(data["precipIntensityMax"]) == 0):
		file.write("{},".format("NA"))
	else:
		file.write("{},".format(sum(data["precipIntensityMax"])/len(data["precipIntensityMax"])))


	if(len(data["temperatureHigh"]) == 0):
		file.write("{},".format("NA"))
	else:
		file.write("{},".format(sum(data["temperatureHigh"])/len(data["temperatureHigh"])))

	if(len(data["temperatureLow"]) == 0):
		file.write("{},".format("NA"))
	else:
		file.write("{},".format(sum(data["temperatureLow"])/len(data["temperatureLow"])))

	if(len(data["humidity"]) == 0):
		file.write("{},".format("NA"))
	else:
		file.write("{}\n".format(sum(data["humidity"])/len(data["humidity"])))



for line in open(filename, "r"):

	if(header is None):
		header = line.split(",")
	else:

		data = line.split(",")

		if(coordinates is None):
			coordinates = data[0]

		if(data[1] == startDate):
			startParse = True 

		
		if(startParse):
			# convert to int:
			for i in range(3,len(data)):
				if(data[i] != "NA" and data[i] != "NA\n"):
					data[i] = float(data[i])


			dictData["precipProbability"].append(data[3])
			dictData["precipIntensity"].append(data[4])
			dictData["precipIntensityMax"].append(data[5])
			dictData["temperatureHigh"].append(data[6])
			dictData["temperatureLow"].append(data[7])
			dictData["humidity"].append(data[8])

			if(day < 6):
				day += 1
			else:
				writeToFile(output, dictData, data[0], data[1])
				dictData = {"precipProbability": [],
					"precipIntensity": [],
					"precipIntensityMax": [],
					"temperatureHigh": [],
					"temperatureLow": [],
					"humidity": []}
				day = 0