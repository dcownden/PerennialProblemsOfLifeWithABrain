import os
import sys
import yaml
from jinja2 import Template
import traceback
import json
from bs4 import BeautifulSoup

REPO = os.environ.get("PPLWB_REPO", "PerennialProblemsOfLifeWithABrain")
ARG = sys.argv[1]


def main():
    with open('sequences/materials.yml') as fh:
        materials = yaml.load(fh, Loader=yaml.FullLoader)

    # Make the dictionary that contains the chapters
    toc = {}
    for m in materials:
        if m['category'] not in toc.keys():
            toc[m['category']] = {'part': m['category'], 'chapters': []}
    # Add the project booklet
    #toc['Project Booklet'] = {'part': 'Project Booklet', 'chapters': []}

    art_file_list = os.listdir('sequences/Art/')

    for m in materials:
        directory = f"{m['chapter']}_{''.join(m['name'].split())}"
        chapter_notebook_name = f"{m['chapter']}_{''.join(m['name'].split())}_Title.ipynb"
        chapter_notebook_path = f"{directory}/{ARG}/{chapter_notebook_name}"

        # Make temporary chapter title file
        if os.path.exists(chapter_notebook_path):
            chapter_content_path = chapter_notebook_path
        else:
            placeholder_content = f"# {m['name']}\n\nChapter overview to go here."
            with open(f"sequences/{directory}/chapter_title.md", "w+") as title_file:
                title_file.write(placeholder_content)
            chapter_content_path = f"sequences/{directory}/chapter_title.md"


        chapter = {
            'file': chapter_content_path,
            'title': f"{m['name']} ({m['chapter']})",
            'sections': []
        }

        print(m['chapter'])
        part = m['category']
        
        # Make list of notebook sections
        notebook_list = []
        # intro and outro stuff moved into _Title.ipynb files for nicer toc
        # notebook_list += [f"{directory}/{ARG}/{m['chapter']}_Intro.ipynb"] if os.path.exists(f"{directory}/{m['chapter']}_Intro.ipynb") else []
        notebook_list += [f"{directory}/{ARG}/{m['chapter']}_Sequence{i + 1}.ipynb" for i m['sequences']]
        # notebook_list += [f"{directory}/{ARG}/{m['chapter']}_Outro.ipynb"] if os.path.exists(f"{directory}/{m['chapter']}_Outro.ipynb") else []

        # Add and process all notebooks
        for notebook_file_path in notebook_list:
            chapter['sections'].append({'file': notebook_file_path})
            pre_process_notebook(notebook_file_path)

        # Add further reading page
        # chapter['sections'].append({'file': f"{directory}/further_reading.md"})

        # Add chapter summary page
        # notebook_file_path = f"{directory}/{ARG}/{m['chapter']}_ChapterSummary.ipynb"
        # if os.path.exists(notebook_file_path):
            # chapter['sections'].append({'file': notebook_file_path})
            # pre_process_notebook(notebook_file_path)

        # Add chapter
        toc[part]['chapters'].append(chapter)

    
    # Turn toc into list
    # there needs to be something with file as a key that is the intro
    # this is an esoteric requirement of the current juptyer-book setup, but we can live with it
    toc_list = [{'file': f"sequences/intro.ipynb"}]
    if os.path.exists(f"sequences/intro.ipynb"):
        pre_process_notebook(f"sequences/intro.ipynb")

    for key in toc.keys():
        # Add wrap-up if it exists
        #wrapup_name = f'sequences/Module_WrapUps/{key.replace(" ", "")}.ipynb'
        #if os.path.exists(wrapup_name):
        #    toc[key]['chapters'].append({'file': wrapup_name})
        toc_list.append(toc[key])

    with open('book/_toc.yml', 'w') as fh:
        yaml.dump(toc_list, fh)

def pre_process_notebook(file_path):

    with open(file_path, encoding="utf-8") as read_notebook:
        content = json.load(read_notebook)
    pre_processed_content = open_in_colab_new_tab(content)
    pre_processed_content = change_video_widths(pre_processed_content)
    pre_processed_content = link_hidden_cells(pre_processed_content)
    with open(file_path, "w", encoding="utf-8") as write_notebook:
        json.dump(pre_processed_content, write_notebook, indent=1, ensure_ascii=False)


def open_in_colab_new_tab(content):
    cells = content['cells']
    parsed_html = BeautifulSoup(cells[0]['source'][0], "html.parser")
    for anchor in parsed_html.findAll('a'):
        # Open in new tab
        anchor['target'] = '_blank'
    cells[0]['source'][0] = str(parsed_html)
    return content

def link_hidden_cells(content):
    cells = content['cells']
    updated_cells = cells.copy()

    i_updated_cell = 0
    for i_cell, cell in enumerate(cells):
        updated_cell = updated_cells[i_updated_cell]
        if "source" not in cell:
            continue
        source = cell['source'][0]

        if source.startswith("#") and cell['cell_type'] == 'markdown':
            header_level = source.count('#')
        elif source.startswith("---") and cell['cell_type'] == 'markdown':
            if len(cell['source']) > 1 and cell['source'][1].startswith("#") and cell['cell_type'] == 'markdown':
                header_level = cell['source'][1].count('#')

        if '@title' in source or '@markdown' in source:
            if 'metadata' not in cell:
                updated_cell['metadata'] = {}
            if 'tags' not in cell['metadata']:
                updated_cell['metadata']['tags'] = []

            # Check if cell is video one
            if 'YouTubeVideo' in ''.join(cell['source']) or 'IFrame' in ''.join(cell['source']):
                if "remove-input" not in cell['metadata']['tags']:
                    updated_cell['metadata']['tags'].append("remove-input")
            else:
                if "hide-input" not in cell['metadata']['tags']:
                    updated_cell['metadata']['tags'].append("hide-input")

            # If header is lost, create one in markdown
            if '@title' in source:

                if source.split('@title')[1] != '':
                    header_cell = {
                        'cell_type': 'markdown',
                        'metadata': {},
                        'source': ['#'*(header_level + 1) + ' ' + source.split('@title')[1]]}
                    updated_cells.insert(i_updated_cell, header_cell)
                    i_updated_cell += 1

            strings_with_markdown = [(i, string) for i, string in enumerate(cell['source']) if '@markdown' in string]
            if len(strings_with_markdown) == 1:
                i = strings_with_markdown[0][0]
                if cell['source'][i].split('@markdown')[1] != '':
                    header_cell = {
                        'cell_type': 'markdown',
                        'metadata': {},
                        'source': [cell['source'][i].split('@markdown')[1]]}
                    updated_cells.insert(i_updated_cell, header_cell)
                    i_updated_cell += 1

        i_updated_cell += 1

    content['cells'] = updated_cells
    return content

def change_video_widths(content):

    for cell in content['cells']:
        if 'YouTubeVideo' in ''.join(cell['source']):

            for ind in range(len(cell['source'])):
                # Change sizes
                cell['source'][ind] = cell['source'][ind].replace('854', '730')
                cell['source'][ind] = cell['source'][ind].replace('480', '410')

        # Put slides in ipywidget so they don't overlap margin
        if len(cell['source']) > 1 and 'IFrame' in cell['source'][1]:
            slide_link = ''.join(cell['source']).split('"')[1].split(", width")[0][:-1]
            cell['source'] = ['# @markdown\n',
                              'from IPython.display import IFrame\n',
                              'from ipywidgets import widgets\n',
                              'out = widgets.Output()\n',
                              'with out:\n',
                              f'    display(IFrame(src=f"{slide_link}", width=730, height=410))\n',
                              'display(out)']
    return content

if __name__ == '__main__':
    main()
