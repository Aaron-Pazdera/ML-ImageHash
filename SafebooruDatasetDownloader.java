import java.io.IOException;
import java.net.URL;
import java.net.URLConnection;
import java.util.ArrayList;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;

public class SafebooruDatasetDownloader {

	public static void main(String[] args) {

		SafebooruDatasetDownloader scraper = new SafebooruDatasetDownloader();

		final int numThreads = 32;
		final int[] pageRequest = new int[numThreads];
		for (int i = 1; i <= numThreads; i++) {
			pageRequest[i - 1] = i;
		}

		class DownloadTrack implements Runnable {
			int i;

			DownloadTrack(int i) {
				this.i = i;
			}

			@Override
			public void run() {
				List<String> scrapedLinks = null;
				for (;;) {
					scrapedLinks = scraper.requestPage(pageRequest[this.i]);
					if (scrapedLinks == null) return;
					pageRequest[this.i] += numThreads;

					synchronized (scraper) { // Is not explicitly thread-safe in the documentation.
						for (String l : scrapedLinks) System.out.println(l);
					}
				}
			}
		}

		ExecutorService pool = Executors.newWorkStealingPool(numThreads);
		for (int i = 0; i < numThreads; i++) {
			pool.execute(new DownloadTrack(i));
		}

		try {
			pool.shutdown();
			pool.awaitTermination(5, TimeUnit.HOURS);
		} catch (InterruptedException e) {
		}
	}

	private static String apiRequestURL = "https://safebooru.org/index.php?page=dapi&s=post&q=index&limit=1000&pid=";

	private SafebooruDatasetDownloader() {
	}

	private List<String> requestPage(int page) {
		try {
			// Load the page requested
			DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
			DocumentBuilder db = dbf.newDocumentBuilder();
			URLConnection connection = new URL(apiRequestURL + page).openConnection();
			connection.setRequestProperty("User-Agent",
					"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_5) AppleWebKit/537.31 (KHTML, like Gecko) Chrome/26.0.1410.65 Safari/537.31");
			Document document = db.parse(connection.getInputStream());

			Element root = document.getDocumentElement();
			if (root.getAttribute("count") == null) return null;

			List<String> toAdd = new ArrayList<>();

			// Rip all the urls from the DOM tree, and add them to the list.
			Element post;
			NodeList posts = root.getChildNodes();
			for (int i = 0; i < posts.getLength(); i++) {
				// This gets the url of the post, which is what we care about, but ignores
				// things like comments/whitespace, and other things we don't care about.
				Node n = posts.item(i);
				if (n.getNodeType() == Node.ELEMENT_NODE) {
					post = (Element) n;
					toAdd.add(post.getAttribute("preview_url"));
				}
			}

			return toAdd;

		} catch (NoSuchElementException e) {
			throw e; // Exception as control flow. Yeet. But it's actually an exceptional case,
						// because you never expect to reach the end of such a massive archive. You'd
						// have to download for multiple days or weeks. We can get away with doing this,
						// because the performance cost is insignificant in terms of days.
		}
		// The logic of the method is finished, handling errors gets a little bit
		// unruly.
		catch (IOException e) {
			// If windows is screwing up, try again. I got these when I closed my laptop.
			if (e instanceof java.net.UnknownHostException) {
				try {
					Thread.sleep(5000L);
				} catch (InterruptedException e1) {
				}
				return this.requestPage(page);
			} else if (e instanceof java.net.SocketException) {
				// This is indicative of a connection issue unrelated to not having Internet.
				return this.requestPage(page);
			}
			// If you get an error in the 500 range, wait 2 seconds and try again.
			else if (e.getMessage().matches(".*5.. for URL.*")) {
				try {
					Thread.sleep(2000);
					return this.requestPage(page);
				} catch (InterruptedException e2) {
					System.err.println(e2.getMessage());
				}

			} else {
				// If it isn't one of these things, at least for the purposes of debugging, I
				// want to make it explode.

				// In this way, this method is guaranteed to somehow work, overflow the
				// function stack, or otherwise halt the entire program.
				System.err.println("An unknown error has occurred. "
						+ "Please create an issue at https://github.com/Aaron-Pazdera/Open-Image-Hashing-Tools "
						+ "with the following stack trace:");
				e.printStackTrace();
				System.exit(1);
			}
		}
		// These two I assume to be unrecoverable and only reachable by programmer
		// error.
		catch (ParserConfigurationException e) {
			System.err.println("Unrecoverable error configuring parser.");
			e.printStackTrace();
			System.exit(2);
		} catch (SAXException e) {
			System.err.println("Unrecoverable error parsing SAX DOM tree.");
			e.printStackTrace();
			System.exit(2);
		}
		return null;
	}
}
