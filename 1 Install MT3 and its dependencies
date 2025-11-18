# Copyright 2021 Google LLC. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#@title Setup Environment
#@markdown Install MT3 and its dependencies (may take a few minutes).

!apt-get update -qq && apt-get install -qq libfluidsynth3 build-essential libasound2-dev libjack-dev

# install mt3
!git clone --branch=main https://github.com/magenta/mt3
!mv mt3 mt3_tmp; mv mt3_tmp/* .; rm -r mt3_tmp
!python3 -m pip install jax[cuda12] nest-asyncio pyfluidsynth==1.3.0 -e . -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# copy checkpoints
!gsutil -q -m cp -r gs://mt3/checkpoints .

# copy soundfont (originally from https://sites.google.com/site/soundfonts4u)
!gsutil -q -m cp gs://magentadata/soundfonts/SGM-v2.01-Sal-Guit-Bass-V1.3.sf2 .

import json
import IPython

# The below functions (load_gtag and log_event) handle Google Analytics event
# logging. The logging is anonymous and stores only very basic statistics of the
# audio and transcription e.g. length of audio, number of transcribed notes.

def load_gtag():
  """Loads gtag.js."""
  # Note: gtag.js MUST be loaded in the same cell execution as the one doing
  # synthesis. It does NOT persist across cell executions!
  html_code = '''
<!-- Global site tag (gtag.js) - Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-4P250YRJ08"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'G-4P250YRJ08',
       {'referrer': document.referrer.split('?')[0],
        'anonymize_ip': true,
        'page_title': '',
        'page_referrer': '',
        'cookie_prefix': 'magenta',
        'cookie_domain': 'auto',
        'cookie_expires': 0,
        'cookie_flags': 'SameSite=None;Secure'});
</script>
'''
  IPython.display.display(IPython.display.HTML(html_code))

def log_event(event_name, event_details):
  """Log event with name and details dictionary."""
  details_json = json.dumps(event_details)
  js_string = "gtag('event', '%s', %s);" % (event_name, details_json)
  IPython.display.display(IPython.display.Javascript(js_string))

load_gtag()
log_event('setupComplete', {})
