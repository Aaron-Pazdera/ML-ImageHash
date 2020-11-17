# ML-ImageHash

A PyTorch implementation of a machine learning perceptual image hash algorithm for near-duplicate detection and fast content-based image retrieval.

Introduces an algorithm which improves upon Triplet Margin Loss as introduced by the FaceNet paper: <https://arxiv.org/abs/1503.03832>

### Dataset download instructions:

``` 
    # Install Java.
    # sudo apt-get install -y default-jdk

    # Enter into the project directory

    # Compile the java code
    ./Compile_Jars.sh

    # Run the link scraper, dumping the links into a text file. This takes 
    # quite a while, depending on internet speeds. I'd recommend running it overnight.
    touch Downloaded_Links.txt
    java -jar Link_Scraper.jar > Downloaded_Links.txt
    
    # Compile and run the image downloader, pointing it at the text file. This 
    # takes way longer. Several days, in fact.
    java -jar Image_Downloader.jar Downloaded_Links.txt
```

### Model training instructions

``` 
    # Install python3-pip. For example, on Debian:
    # sudo apt-get install -y python3-pip

    # Install Jupyter Notebooks.

    # Install the required pip packages. This amounts to resolving import errors as they come up.

    # Hit run all. It will train the model, then export it into the Models folder, and then try to test it.
```

### Model testing instructions

At the end of the notebook, there's some code that visually verifies the model using nearest neighbor lookup. The data structure used for this lookup is my implementation of a Vantage Point Tree, which can be found below. Install it, and make sure the bindings are in your PYTHONPATH.

<https://github.com/apaz-cli/VPTree>
