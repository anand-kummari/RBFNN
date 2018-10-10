f=open("stock_output.txt","r")
count=0
sum=0
for line in f:
	sum=sum+float(line)
	count=count+1
avg=sum/count
f.close()
file=open("testfile.txt","w")
f=open("stock_output.txt","r")

for line in f:
	# print("anand")
	# print(float(line))
	if float(line) < avg:
		file.write("0\n")
		# print("if\n")
	else:
		file.write("1\n")
