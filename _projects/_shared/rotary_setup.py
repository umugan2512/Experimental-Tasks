# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Rotary encoder connection/setup boilerplate, factored out of wheel_turn_reward.py/
hifi_singleside_test.py/hifi_alternating_easy_test.py where it used to be copy-pasted. See
bpod_trial_helpers.py's module docstring for why this lives at _projects/_shared/ rather than
under any one project's tasks/ folder.
"""
from pybpod_rotaryencoder_module.module_api import RotaryEncoderModule
from pybpod_rotaryencoder_module.module import RotaryEncoder


def connect_rotary(bpod, usb_port=None):
    """ Resolves the Bpod-relay-side module (for building relay triggers) and connects the
    rotary's own direct-USB API object (for current_position()/set_thresholds()/etc.) -- these are
    two separate physical connections, never read off each other (see CLAUDE.md). Returns
    (rotary, rotary_bpod_module). """
    rotary_bpod_module = next(m for m in bpod.modules if m.name and m.name.startswith('RotaryEncoder'))
    print("Resolved module '{0}' on Bpod port {1}".format(
        rotary_bpod_module.name, rotary_bpod_module.serial_port), flush=True)

    if usb_port:
        rotary = RotaryEncoderModule(usb_port)
    else:
        rotary = RotaryEncoderModule.discover(exclude_ports=[bpod.serial_port])
    print("Connected to rotary encoder on {0}".format(rotary.arcom.serial_object.port), flush=True)

    return rotary, rotary_bpod_module


def build_reset_trigger(rotary_bpod_module):
    """
    Bundles SETZEROPOS+ENABLE_ALLTHRESHOLDS into a single relay message, for use as an
    output_actions entry (e.g. fired on WaitForChoice's own entry so left/right are measured fresh
    from wherever the wheel rests the instant a choice window opens).

    NOT rotary_bpod_module.create_resetpositions_trigger() -- that convenience method only exists
    on the custom RotaryEncoder subclass, and module classification depends on PYBPOD_API_MODULES
    being set when BpodModules.create_module() first runs. Each session folder gets its own
    auto-generated user_settings.py (missing PYBPOD_API_MODULES entirely), and a task runs as a
    subprocess with cwd set to that session folder -- confirmed on hardware: rotary_bpod_module
    resolves to a plain BpodModule there, not RotaryEncoder, so calling
    create_resetpositions_trigger() on it raises AttributeError. load_message() is on the
    BpodModule base class itself, so the same message is built manually here from RotaryEncoder's
    own command byte constants (importable directly from the class, no instance/classification
    needed).

    :return: (trigger_id, channel) -- pass as (channel, trigger_id) in output_actions.
    """
    trigger_id = rotary_bpod_module.load_message(
        [RotaryEncoder.COM_SETZEROPOS, RotaryEncoder.COM_ENABLE_ALLTHRESHOLDS])
    channel = 'Serial{0}'.format(rotary_bpod_module.serial_port)
    return trigger_id, channel


def set_and_enable_thresholds(rotary, thresholds_deg):
    """
    Sets and arms every threshold in thresholds_deg (max 6, per set_thresholds()'s own limit), and
    returns the resulting Bpod event name for each, in the same order -- e.g. thresholds_deg[0]'s
    crossing event is the first string returned. Centralizes the "list index -> event number"
    mapping in one place (RotaryEncoder1_<index+1>) instead of every task script re-deriving it by
    hand from a threshold list built alongside separately-named constants.
    """
    rotary.set_thresholds(thresholds_deg)
    rotary.enable_all_thresholds()
    return ['RotaryEncoder1_{0}'.format(i + 1) for i in range(len(thresholds_deg))]
