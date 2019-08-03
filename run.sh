
if [ $# -gt 0 ]; then
    echo "Your command line contains $# arguments"
else
    echo "Your command line contains no arguments"
fi

if [ "$1" == "-recompf" ]; then
    echo "compile fortran"
    g++ -c specfun.f -O3
fi

echo "compile main"
g++ -g -c main.cpp -O3

echo "build main"
g++ -o main main.o specfun.o -lgfortran -llapack -llapacke -lopenblas
#g++ -o main libblas.a liblapack.a main.o specfun.o -lgfortran -llapacke 

echo "run main"
./main

