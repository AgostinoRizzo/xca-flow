import re
import sys

coord = [100,100,50]

nameCoord = ["X_COORDINATES","Y_COORDINATES","Z_COORDINATES"]

user_args = sys.argv[1:]

for input in user_args:
    input2 = input[:-4]
    print(input2)

    fileoutput = open(input2+".vtk", "w")
    fileoutput.write("# vtk DataFile Version 2.0\n")
    fileoutput.write("Sample rectilinear grid\n")
    fileoutput.write("ASCII\n")
    fileoutput.write("DATASET RECTILINEAR_GRID\n")
    fileoutput.write("DIMENSIONS "+str(coord[0])+" "+str(coord[1])+" "+str(coord[2])+"\n") 
    
    for c,n in zip(coord,nameCoord):
        fileoutput.write(str(n)+" "+ str(c) + " float\n")
        increment = 0
        for x in range(0,c,1):
            fileoutput.write(str(increment) +" ")
            increment+=1
        fileoutput.write("\n")
    
    fileoutput.write("POINT_DATA "+ str(coord[0]*coord[1]*coord[2]) + "\n")
    fileoutput.write("SCALARS scalars float\n")
    fileoutput.write("LOOKUP_TABLE default\n")
    
    fileread = open(input, "r")
    
    line = fileread.readline()
    cnt = 1
    
    while line:
        splitted = line.split("\t")
        #print(splitted)
        count = 1
        for s in splitted:
            s = re.sub("\n","",s)
           # s = re.sub(" ",";",s)
            fileoutput.write(s)
            if(count%5 == 0):
                fileoutput.write("\n")
            else:
                fileoutput.write(" ")
            count+=1
        line = fileread.readline()
        cnt += 1
    
    fileoutput.close()
