import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import logging
import openai
from flask import Flask, request

from CallingGPT.src.CallingGPT.session.session import GPT_Session_Handler
from autonomous_om import om_func_list
from autonomous_om.om_prompt import SYSTEM_PROMPT_OM_AGENT, INIT_PROMPT_OM_AGENT
from utils.sensitives import OPENAI_API_KEY

app = Flask(__name__)

# log.settings
app.logger.setLevel(logging.WARNING)
log = logging.getLogger('werkzeug')
log.setLevel(logging.WARNING)

prompt_ending = ""


class OM_GPT_Actuator(GPT_Session_Handler):
    """
    this class is used to store a separate om actuator session, which can iterate with the information provided by the web chat session with vision
    """
    pass


@app.route('/')
def status_check():
    return 'the server is running ok!', 200


@app.route('/om_imaging', methods=['POST'])
def om_imaging():

    # initialize sem agent session
    openai.api_key = OPENAI_API_KEY
    sem_agent = OM_GPT_Actuator.get_instance(
        modules=[om_func_list],
        model="gpt-4o-mini-2024-07-18",
        system_prompt=SYSTEM_PROMPT_OM_AGENT + request.get_json()['final_objective'],
        temperature=0,
    )
    final_res = sem_agent.ask(INIT_PROMPT_OM_AGENT)
    return final_res, 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8050, debug=True)
