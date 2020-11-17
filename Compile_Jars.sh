#!/usr/bin/env bash

# Pack the Scraper java file into a jar
javac SafebooruDatasetDownloader.java
echo 'Main-Class: SafebooruDatasetDownloader' > manifest.mf
jar -cvmf manifest.mf Link_Scraper.jar *.class > /dev/null

# Clean up generated class files and manifest
rm 'SafebooruDatasetDownloader.class'
rm 'SafebooruDatasetDownloader$1DownloadTrack.class'
rm 'manifest.mf'

# Pack the Image Downloader java file into a jar
javac DownloadLinks.java
echo 'Main-Class: DownloadLinks' > manifest.mf
jar -cvmf manifest.mf Image_Downloader.jar *.class > /dev/null

# Clean up
rm 'DownloadLinks.class'
rm 'manifest.mf'