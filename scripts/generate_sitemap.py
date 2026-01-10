import requests
from datetime import datetime
import xml.etree.ElementTree as ET
from xml.dom import minidom
import os

# Ghost API settings
GHOST_CONTENT_API_KEY = os.getenv(
    "GHOST_CONTENT_API_KEY"
)  # Replace with your Content API key
DOMAIN = "https://oksayless.com"
GHOST_URL = "https://always-be-optimizing.ghost.io"  # Update with your Ghost URL
API_VERSION = "v5.0"


def fetch_posts():
    url = f"{GHOST_URL}/ghost/api/content/posts/"
    params = {"key": GHOST_CONTENT_API_KEY, "fields": "slug,updated_at", "limit": "all"}

    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()["posts"]
    else:
        raise Exception(f"Failed to fetch posts: {response.status_code}")


def generate_sitemap():
    # Create the root element
    urlset = ET.Element("urlset")
    urlset.set("xmlns", "http://www.sitemaps.org/schemas/sitemap/0.9")

    # Add homepage
    home_url = ET.SubElement(urlset, "url")
    ET.SubElement(home_url, "loc").text = DOMAIN
    ET.SubElement(home_url, "changefreq").text = "monthly"
    ET.SubElement(home_url, "priority").text = "1.0"

    # Add blog index
    blog_url = ET.SubElement(urlset, "url")
    ET.SubElement(blog_url, "loc").text = f"{DOMAIN}/blog"
    ET.SubElement(blog_url, "changefreq").text = "daily"
    ET.SubElement(blog_url, "priority").text = "0.8"

    # Add all blog posts
    posts = fetch_posts()
    for post in posts:
        url = ET.SubElement(urlset, "url")
        ET.SubElement(url, "loc").text = f"{DOMAIN}/blog/{post['slug']}"
        ET.SubElement(url, "lastmod").text = post["updated_at"][
            :10
        ]  # YYYY-MM-DD format
        ET.SubElement(url, "changefreq").text = "never"
        ET.SubElement(url, "priority").text = "0.6"

    # Create the XML string with pretty printing
    xmlstr = minidom.parseString(ET.tostring(urlset)).toprettyxml(indent="   ")

    # Write to file
    with open("frontend/sitemap.xml", "w", encoding="utf-8") as f:
        f.write(xmlstr)


if __name__ == "__main__":
    generate_sitemap()
