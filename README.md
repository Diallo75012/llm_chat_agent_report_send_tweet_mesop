### UI APP For LLM interactions using Python
This app will:
- generate answers to user chat queries.
- have a sidebar for user to enter tweeter tokens
- use agents to create a tweet based on the last chat answer on the fly
- have extra button to generate a report made by agents on last answer from the chat and will download the report in user local computer

# preriquisites:
- tweeter consumer and access: token and secrets

# Mesop from google instead of streamlit/gradio
# easy components setup
# easy management of state variables (that is why mesop has been chosen)

# clone
```bash
git clone
```

# create virtual env
```python
python3 -m venv <name_of_your_virtual_environment>
```

# install dependencies
```python
pip install -r requirements.txt
```

# start the app (feel free to modify it and play with the UI components to get a hand of it and build your own logic)
```python
mesop app.py
```

