import sys
import string
import math

if __name__ == "__main__":
	inputfile = "data_original.tsv"
	o1 = "traindata.tsv"
	o2 = "validationdata.tsv"
	f = open(inputfile,"r")
	fo1 = open(o1,"w+")
	fo2 = open(o2,"w+")
	no_of_lines = len(f.readlines())
	threshold = math.floor(no_of_lines*0.8)
	i=1
	f.seek(0,0)
	for line in f:
		if i <= threshold:
			fo1.write(line)
		else:
			fo2.write(line)
		i +=1
		#print(i)

	fo1.close()
	fo2.close()


	



