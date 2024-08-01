chart2text_templates = {
    'image':{
        '0':'The columns of the chart:\n{columns}\nSummarize the chart in a few sentences.',
        '1':'Summarize the following chart about {columns} in a few sentences:\n',
        '2':'Summarize the following chart titled "{title}" about "{columns}" in a few sentences:\n',
        '3':'Generate a caption that describes the statistic below. The caption should explain  “{columns}”. The caption should be placed between  <caption>output</caption> tags.\nThe tile of the input statistic:\n{title}\nOutput:\n',
        '4':'Title of the statistic:\n{title}\nAxis label of the statistic:\n{columns}\nGenerate a caption that describes the statistic. The caption should be placed between  <caption>output</caption> tags.\nOutput:\n',
        '5':'Title of the statistic:\n{title}\nAxis label of the statistic:\n{columns}\nGenerate a caption that describes the statistic.',
        '6':'Title of the statistic:\n{title}\nColumn of the statistic:\n{columns}\nGenerate a caption that describes the statistic.',
    },
    'text':{
        '0':'The columns of the table:\n{columns}\nTable:\n{table}\nSummarize the table in a few sentences.',
        '1':'Summarize the following table about {columns} in a few sentences:\n{table}\n',
        '2':'Summarize the following table titled "{title}" about "{columns}" in a few sentences:\n{table}\n',
        '3':'Generate a caption that describes the statistic below. The caption should explain  “{columns}”. The caption should be placed between  <caption>output</caption> tags.\nThe tile of the input statistic:\n{title}\nThe content of the statistic:\n{table}\nOutput:\n',
        '4':'Title of the statistic:\n{title}\nAxis label of the statistic:\n{columns}\nContent of the statistic:\n{table}\nGenerate a caption that describes the statistic. The caption should be placed between  <caption>output</caption> tags.\nOutput:\n',
        '5':'Title of the statistic:\n{title}\nAxis label of the statistic:\n{columns}\nContent of the statistic:\n{table}\nGenerate a caption that describes the statistic.',
        '6':'Title of the statistic:\n{title}\nColumns of the statistic:\n{columns}\nStatistic:\n{table}\nGenerate a caption that describes the statistic.',
    },
    'image_text':{
        '0':'The columns of the table and chart:\n{columns}\nTable:\n{table}\nSummarize the chart and table in a few sentences.',
        '1':'Summarize the following table and chart about {columns} in a few sentences:\n{table}\n',
        '2':'Summarize the following table and chart titled "{title}" about "{columns}" in a few sentences:\n{table}\n',
        '3':'Generate a caption that describes the statistic below. The caption should explain  “{columns}”. The caption should be placed between  <caption>output</caption> tags.\nThe tile of the input statistic:\n{title}\nThe content of the statistic:\n{table}\nOutput:\n',
        '4':'Title of the statistic:\n{title}\nAxis label of the statistic:\n{columns}\nContent of the statistic:\n{table}\nGenerate a caption that describes the statistic. The caption should be placed between  <caption>output</caption> tags.\nOutput:\n',
        '5':'Title of the statistic:\n{title}\nAxis label of the statistic:\n{columns}\nContent of the statistic:\n{table}\nGenerate a caption that describes the statistic.',
        '6':'Title of the statistic:\n{title}\nColumns of the statistic:\n{columns}\nStatistic:\n{table}\nGenerate a caption that describes the statistic.',

    },
    'bridge':{
        '0':'',
        '1':'Summarize the following table and chart about {columns} in a few sentences:\n{table}\nThis Chart is based on the table.\n',
        '2':'Summarize the following table and chart about {columns} in a few sentences:\n{table}\n{draft}\n',
        '3':'Summarize the following table and chart titled "{title}" about "{columns}" in a few sentences:\n{table}\nThis Chart is based on the table.\n',
        '4':'Summarize the following table and chart titled "{title}" about "{columns}" in a few sentences:\n{table}\n{draft}\n',
        '5':'Generate a caption that describes the statistic below. The caption should explain  “{columns}”. The caption should be placed between  <caption>output</caption> tags.\nThe tile of the input statistic:\n{title}\nThe content of the statistic:\n{table}\nThis Chart is based on the statistic.\nOutput:\n',
        '6':'Generate a caption that describes the statistic below. The caption should explain  “{columns}”. The caption should be placed between  <caption>output</caption> tags.\nThe tile of the input statistic:\n{title}\nThe content of the statistic:\n{table}\n{draft}\nOutput:\n',
        # 'code':'Title of the statistic:\n{title}\nAxis label of the statistic:\n{columns}\nContent of the statistic:\n{table}\nThe code which makes the image:\n{draft}\nGenerate a caption that describes the statistic. The caption should be placed between  <caption>output</caption> tags.\nOutput:\n',
        'code':'Title of the statistic:\n{title}\nAxis label of the statistic:\n{columns}\nContent of the statistic:\n{table}\nThe code which makes the image:\n{draft}\nGenerate a caption that describes the statistic.',
        # 'auto-cot':'Title of the statistic:\n{title}\nAxis label of the statistic:\n{columns}\nContent of the statistic:\n{table}\nThis Chart is based on the statistic.\nGenerate a caption that describes the statistic.',
        'auto-cot':'Title of the statistic:\n{title}\nColumns of the statistic:\n{columns}\nStatistic:\n{table}\nThis Chart is based on the statistic.\nGenerate a caption that describes the statistic.',
        # 'auto-cot':'Title of the statistic:\n{title}\nAxis label of the statistic:\n{columns}\nContent of the statistic:\n{table}\nThis Chart is based on the statistic.\nGenerate a caption that describes the statistic. The caption should be placed between  <caption>output</caption> tags.\nOutput:\n',
        # 'base':'Title of the statistic:\n{title}\nAxis label of the statistic:\n{columns}\nContent of the statistic:\n{table}\n{draft}\nGenerate a caption that describes the statistic. The caption should be placed between  <caption>output</caption> tags.\nOutput:\n',
        'base':'Title of the statistic:\n{title}\nAxis label of the statistic:\n{columns}\nContent of the statistic:\n{table}\n{draft}\nGenerate a caption that describes the statistic.',
        
    }
}

chartqa_templates = {

}

def get_template(
        task:str,
        inputs:str,
        template_number:str,
):
    templates = chart2text_templates if task == 'chart2text' else chartqa_templates
    return templates[inputs][template_number]