# GGML for MLP
Create model and convert it to GGML format.
```
python model.py
python convert.py
```

Build and run the example.
```
cd ~ # go to home directory
rm -r -f build # remove the build directory
cmake -B build # set the build directory 
cmake --build . --config Release -j16 # build project
./build/bin/alex-mlp ~/llama.cpp/examples/alex-mlp/model/mlp.gguf # Run the example
```
