import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import javax.imageio.ImageIO;

public class DownloadLinks {

	private static int numThreads = 32;

	private static File trainFolder;
	private static List<String> trainURLs = new ArrayList<>();
	private static AtomicInteger trainNum = new AtomicInteger(0);

	public static void main(String[] args) {
		// Figure out where to put the images.
		{
			File cwd = new File(System.getProperty("user.dir"));
			trainFolder = new File(cwd, "Dataset");
			if (!trainFolder.isDirectory()) {
				System.err.println(
						"Couldn't find the Dataset folder. Please execute this from the project folder, which contains the Dataset folder.");
				System.exit(0);
			}
		}

		// Load the links into memory and split them into train and valid datasets.
		{
			if (args.length != 1) {
				System.err.println("Please provide exactly one argument, the file of links to download.");
				System.exit(0);
			}
			File linkFile = new File(args[0]);

			Scanner reader = null;
			try {
				reader = new Scanner(linkFile);
			} catch (FileNotFoundException e) {
				System.err.println("The link file specified does not exist.");
				System.exit(0);
			}

			while (reader.hasNext()) {
				trainURLs.add(reader.nextLine());
			}

			reader.close();
		}

		// Spin up a threadpool and start downloading from the links. Once each image is
		// downloaded, save it.
		{
			ForkJoinPool customThreadPool = new ForkJoinPool(numThreads);
			customThreadPool.submit(() -> trainURLs.parallelStream().forEach(url -> downloadImage(url)));
			trainURLs.forEach(url -> customThreadPool.submit(() -> downloadImage(url)));
			try {
				customThreadPool.shutdown();
				customThreadPool.awaitTermination(14, TimeUnit.DAYS);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
	}

	private static void downloadImage(String strURL) {
		// Convert to url
		URL url = null;
		try {
			url = new URL(strURL);
		} catch (MalformedURLException e) {
			System.err.println("Failed to convert malformed URL: " + strURL);
			return;
		}

		// Download the image
		BufferedImage img = openImage(url);
		if (img == null) {
			System.err.println("Download failed for: " + strURL);
			return;
		}

		// Resize
		img = resizeImage(img, 64, 64);

		// Save the image
		String imgName = trainNum.getAndIncrement() + ".png";
		try {
			ImageIO.write(img, "png", new File(trainFolder, imgName));
		} catch (IOException e) {
			System.err.println("Failed to write image: " + strURL + " to " + imgName);
			e.printStackTrace();
		}
	}

	private static BufferedImage openImage(URL imgURL) {
		try {
			final HttpURLConnection connection = (HttpURLConnection) imgURL.openConnection();
			connection.setRequestProperty("User-Agent",
					"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_5) AppleWebKit/537.31 (KHTML, like Gecko) Chrome/26.0.1410.65 Safari/537.31");

			int code = connection.getResponseCode();
			if (code == 403) {
				System.err.println("Error 403: Forbidden for: " + imgURL);
				return null;
			} else if (code == 404) {
				System.err.println("Error 404: Not Found for: " + imgURL);
				return null;
			} else if (code < 500 && code >= 400) { // 4xx error
				System.err.println("Client error (" + code + ") for: " + imgURL);
				return null;
			} else if (code < 600 && code >= 500) { // 5xx error
				System.err.println("Server error (" + code + ") for: " + imgURL + ". Retrying.");
				return openImage(imgURL);
			}

			InputStream s = new BufferedInputStream(connection.getInputStream());
			return ImageIO.read(s);
		} catch (IOException e) {
			// If something goes wrong, wait a bit and try again.
			try {
				Thread.sleep(5000);
			} catch (InterruptedException e1) {
			}
			return openImage(imgURL);
		} catch (Exception e) {
			// Due to a Java 8 bug in ImageIO, we cannot read animated gifs. The documented
			// behavior is that it's supposed to read the first frame, but instead an array
			// out of bounds exception is thrown from the standard library. It sucks, but
			// there's nothing that can be done.
			return null;
		}
	}

	private static BufferedImage resizeImage(BufferedImage img, int width, int height) {
		BufferedImage resized = new BufferedImage(width, height, img.getType());
		Graphics2D g = resized.createGraphics();
		g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
		g.drawImage(img, 0, 0, width, height, 0, 0, img.getWidth(), img.getHeight(), null);
		g.dispose();

		return resized;
	}

}
