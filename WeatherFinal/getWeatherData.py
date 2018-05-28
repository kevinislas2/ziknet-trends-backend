import requests
import datetime

key = None

# latitude = "19.1500" # Veracruz
# altitude = "-96.1170" # Veracruz

# latitude = -14.8630 # Vitoria da conquista, Bahia, Brazil
# altitude = -40.8630 # Vitoria da conquista, Bahia, Brazil

latitude = "25.7330" #Nuevo León
altitude = "-100.3000" # Nuevo León

baseURL = "https://api.darksky.net/forecast/{}/{},{}".format(key, latitude, altitude)

#iterate on days
dateString = input("Enter first day in DD-MM-YYYY format: ")
dateArray = [int(i) for i in dateString.split("-")]
dateObj = datetime.datetime(dateArray[2],dateArray[1],dateArray[0])

numberOfDays = int(input("Enter number of days to calculate: "))

fileName = "NuevoLeon_{}_{}.csv".format(dateString,numberOfDays)
file = open(fileName, "w")
file.write("coordinates,date,precipType,precipProbability,precipIntensity,precipIntensityMax,temperatureHigh,temperatureLow,humidity\n")
keys = ["precipType","precipProbability","precipIntensity","precipIntensityMax","temperatureHigh","temperatureLow","humidity"]

def writeToFile(dateString, response):
	data = response.json()["daily"]["data"][0]
	file.write("{};{},".format(latitude, altitude))
	file.write("{},".format(dateString))

	for i in range(len(keys)):
		key = keys[i]
		if(key in data):
			file.write(str(data[key]))
		else:
			file.write("NA")
		if i < len(keys)-1:
			file.write(",")
	file.write("\n")

while numberOfDays > 0:

	dateString = "{}-{:02}-{:02}".format(dateObj.year, dateObj.month, dateObj.day)

	url = "{},{}T12:00:00?exclude=hourly&units=si".format(baseURL, dateString)
	
	request = requests.get(url)
	if(request.status_code == 200):
		writeToFile(dateString, request)
	else:
		print("Error")

	dateObj += datetime.timedelta(days=1)
	numberOfDays -= 1