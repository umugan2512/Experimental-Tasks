# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
HiFi module connection/setup boilerplate, factored out of hifi_singleside_test.py/
hifi_alternating_easy_test.py where it used to be copy-pasted. See bpod_trial_helpers.py's module
docstring for why this lives at _projects/_shared/ rather than under any one project's tasks/
folder.
"""
from pybpod_hifi_module.module_api import HiFiModule
from pybpod_hifi_module.module import HiFi, HiFiCommandType


def connect_hifi(bpod, usb_port=None):
    """ Connects the HiFi module's own direct-USB API object (for load()/push()/play()/stop()).
    PLAY is meant to stay on this direct-USB connection rather than a Bpod-relayed trigger -- see
    the calling task script's docstring for why (the same-module-twice-per-trial hang hazard once
    a Bpod-relayed STOP_ALL is also in play). """
    if usb_port:
        hifi = HiFiModule(usb_port)
    else:
        hifi = HiFiModule.discover(exclude_ports=[bpod.serial_port])
    print("Connected to HiFi module on {0} ({1}Hz sampling)".format(
        hifi.arcom.serial_object.port, hifi.sampling_rate), flush=True)
    return hifi


def build_stop_trigger(bpod):
    """
    Registers HiFiCommandType.STOP_ALL as a Bpod-relayed message, for use as an output_actions
    entry (e.g. fired on an abort state's own entry so the cue stops in exact lockstep with Bpod's
    own abort detection, no Python-side hifi.stop() call needed).

    hifi_bpod_module resolves to a plain BpodModule (not the custom HiFi subclass) for the same
    per-session PYBPOD_API_MODULES classification reason documented in rotary_setup.py's
    build_reset_trigger() -- irrelevant here since HiFi.get_command() is a staticmethod needing no
    instance/classification, and load_message() is on the BpodModule base class.

    :return: (stop_msg_id, channel) -- pass as (channel, stop_msg_id) in output_actions.
    """
    hifi_bpod_module = next(m for m in bpod.modules if m.name and m.name.startswith('HiFi'))
    stop_msg_id = hifi_bpod_module.load_message(HiFi.get_command(HiFiCommandType.STOP_ALL))
    channel = 'Serial{0}'.format(hifi_bpod_module.serial_port)
    return stop_msg_id, channel
