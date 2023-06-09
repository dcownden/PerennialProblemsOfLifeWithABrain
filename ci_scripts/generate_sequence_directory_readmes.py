"""Write a directory of sequence notebooks to the README file.

Run this script from the root of the github repository.

"""
import os
from glob import glob
import yaml

REPO = os.environ.get("PPLWB_REPO", "part-content")
MAIN_BRANCH = os.environ.get("PPLWB_MAIN_BRANCH", "main")


def main():

    # Initialize the lines in sequences/README.md
    part_readme_text = [
    ]

    #try:
    #    playlist_urls = load_youtube_playlist_urls()
    #except Exception as err:
    #    print("Encountered error while loading youtube playlist links")
    #    print(err)
    #    playlist_urls = {}

    #try:
    #    slide_urls = load_slide_urls()
    #except Exception as err:
    #    print("Encountered error while loading slide links")
    #    print(err)
    #    slide_urls = {}

    chapter_anchors = {} #part chapter sequence

    chapter_paths = sorted(glob("sequences/P?C?_*"))
    for chapter_path in chapter_paths:

        chapter_name = os.path.split(chapter_path)[-1]
        chapter_code, topic_code = chapter_name.split("_")

        # Split the UpperCamelCase topic name into separate words
        topic_words = []
        for letter in topic_code:
            if letter.isupper():
                topic_words.append(letter)
            else:
                topic_words[-1] += letter
        topic = " ".join(topic_words)

        # Note: this will fail if we have 10+ notebooks
        notebooks = sorted(glob(f"{chapter_path}/*.ipynb"))

        if not notebooks:
            continue

        # Track the anchor to this section for embed in the header
        anchor = "-".join([
            chapter_code.lower(),
            "-",
            ("-".join(topic_words)).lower(),
        ])

        chapter_anchors[chapter_code] = "#" + anchor

        instructor_notebooks = get_instructor_links(notebooks)
        student_notebooks = get_student_links(notebooks)

        # Write the chapter information into the part README
        part_readme_text.extend([
            f"## {chapter_code} - {topic}",
            "",
        ])

        # Add a link to the YouTube lecture playlist, if we have one
        #youtube_url = playlist_urls.get(chapter_code, None)
        #if youtube_url is not None:
        #    part_readme_text.extend([
        #        f"[YouTube Playlist]({youtube_url})"
        #        "",
        #    ])

        #slide_links_by_topic = slide_urls.get(chapter_code, None)
        #if slide_links_by_topic is not None:
        #    slide_links = [
        #        f"[{topic}]({url})" for topic, url in slide_links_by_topic
        #    ]
        #    part_readme_text.extend([
        #        "",
        #        "Slides: " + " | ".join(slide_links),
        #        "",
        #    ])

        part_readme_text.extend(write_badge_table(student_notebooks))
        part_readme_text.append("\n")

        # Add further reading
        further_reading_file = f"{chapter_path}/further_reading.md"
        if os.path.exists(further_reading_file):
            reading_url = f"https://github.com/dcownden/{REPO}/blob/{MAIN_BRANCH}/{further_reading_file}"
            part_readme_text.extend([f"[Further Reading]({reading_url})"])
            part_readme_text.append("\n")

        # Now make the chapter-specific README
        # with links to both instructor and student versions
        chapter_readme_text = [
            f"# {chapter_code} - {topic}",
            "",
            "## Instructor notebooks",
            "",
        ]
        chapter_readme_text.extend(write_badge_table(instructor_notebooks))

        chapter_readme_text.extend([
            "## Student notebooks",
            "",
        ])
        chapter_readme_text.extend(write_badge_table(student_notebooks))

        # Write the chapter README file
        with open(f"{chapter_path}/README.md", "w") as f:
            f.write("\n".join(chapter_readme_text))

    # Create relative anchor links to each chapter
    nav_line = " | ".join([
        f"[{chapter_code}]({anchor})" for chapter_code, anchor in chapter_anchors.items()
    ])

    # Add an introductory header to the main README
    part_readme_header = [
        "# Comp Nuero Book Interactive Sequences",
        "",
        "<!-- DO NOT EDIT THIS FILE. IT IS AUTO-GENERATED BY A FRIENDLY ROBOT -->",
        "",
        nav_line,
        "",
        "*Warning:* The 'render with NBViewer' buttons may show outdated content.",
        "",
    ]
    part_readme_text = part_readme_header + part_readme_text

    # Write the part README file
    with open("sequences/README.md", "w") as f:
        f.write("\n".join(part_readme_text))


def load_youtube_playlist_urls():
    """Create a mapping from chapter code to youtube link based on text file."""
    with open('sequences/materials.yml') as fh:
        materials = yaml.load(fh, Loader=yaml.FullLoader)
    chapters = [m['chapter'] for m in materials]
    playlists = [m['playlist'] for m in materials]
    return dict(zip(chapters, playlists))


def load_slide_urls():
    """Create a hierarchical mapping to slide PDF urls based on text file."""
    with open('sequences/materials.yml') as fh:
        materials = yaml.load(fh, Loader=yaml.FullLoader)
    slide_links = {}
    for chapter_dict in materials:
        if 'slides' in chapter_dict:
            slide_links[chapter_dict['chapter']] = []
            for slide_info in chapter_dict['slides']:
                slide_links[chapter_dict['chapter']].append((slide_info['title'], slide_info['link']))
    return slide_links


def write_badge_table(notebooks):
    """Make a markdown table with colab/nbviewer badge links."""

    # Add the table header
    table_text = [
        "|   | Run | Run | View |",
        "| - | --- | --- | ---- |",
    ]

    # Get ordered list of file names
    notebook_list = [name for name in notebooks if 'Intro' in name]
    notebook_list += [name for name in notebooks if 'Sequence' in name]
    notebook_list += [name for name in notebooks if 'Outro' in name]

    # Add badges
    for local_path in notebook_list:
        # Extract type of file (intro vs outro vs sequence)
        notebook_name = local_path.split('_')[-1].split('.ipynb')[0]

        # Add space between Sequence and number
        if 'Sequence' in notebook_name:
            notebook_name = f"Sequence {notebook_name.split('Sequence')[1]}"
        colab_badge = make_colab_badge(local_path)
        kaggle_badge = make_kaggle_badge(local_path)
        nbviewer_badge = make_nbviewer_badge(local_path)
        table_text.append(
            f"| {notebook_name} | {colab_badge} | {kaggle_badge} | {nbviewer_badge} |"
        )
    table_text.append("\n")

    return table_text


def get_instructor_links(base_notebooks):
    """Convert a list of base notebook paths to instructor versions."""
    instructor_notebooks = []
    for base_nb in base_notebooks:
        if 'Sequence' in base_nb:
            chapter_path, nb_fname = os.path.split(base_nb)
            instructor_notebooks.append(f"{chapter_path}/instructor/{nb_fname}")
        else:
            instructor_notebooks.append(base_nb)
    return instructor_notebooks


def get_student_links(base_notebooks):
    """Convert a list of base notebook paths to student versions."""
    student_notebooks = []
    for base_nb in base_notebooks:
        if 'Sequence' in base_nb:
            chapter_path, nb_fname = os.path.split(base_nb)
            student_notebooks.append(f"{chapter_path}/student/{nb_fname}")
        else:
            student_notebooks.append(base_nb)
    return student_notebooks


def make_colab_badge(local_path):
    """Generate a Google Colaboratory badge for a notebook on github."""
    alt_text = "Open In Colab"
    badge_svg = "https://colab.research.google.com/assets/colab-badge.svg"
    service = "https://colab.research.google.com"
    url_base = f"{service}/github/dcownden/{REPO}/blob/{MAIN_BRANCH}"
    return make_badge(alt_text, badge_svg, service, local_path, url_base)


def make_kaggle_badge(local_path):
    """Generate a kaggle badge for a notebook on github."""
    alt_text = "Open In kaggle"
    badge_svg = "https://kaggle.com/static/images/open-in-kaggle.svg"
    service = "https://kaggle.com/kernels/welcome?src="
    url_base = f"{service}https://raw.githubusercontent.com/dcownden/{REPO}/{MAIN_BRANCH}"
    return make_badge(alt_text, badge_svg, service, local_path, url_base)


def make_nbviewer_badge(local_path):
    """Generate an NBViewer badge for a notebook on github."""
    alt_text = "View the notebook"
    badge_svg = "https://img.shields.io/badge/render-nbviewer-orange.svg"
    service = "https://nbviewer.jupyter.org"
    url_base = f"{service}/github/dcownden/{REPO}/blob/{MAIN_BRANCH}"
    return make_badge(alt_text, badge_svg, service, f"{local_path}?flush_cache=true", url_base)


def make_badge(alt_text, badge_svg, service, local_path, url_base):
    """Generate a markdown element for a badge image that links to a file."""
    return f"[![{alt_text}]({badge_svg})]({url_base}/{local_path})"


if __name__ == "__main__":

    main()
