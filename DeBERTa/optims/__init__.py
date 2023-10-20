#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Author:     (  )
# Date: 05/15/2019
#

""" optimizers
"""

from .args import get_args
from .fp16_optimizer import *
from .lr_schedulers import SCHEDULES
from .xadam import XAdam
