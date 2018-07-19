import igm
import sys
import os.path
from igm.utils.log import set_log, logger
from shutil import copyfile

#===start pipeline with configure file
cfgfile = os.path.abspath(sys.argv[1])
cfg = igm.Config(cfgfile)

logger.info('Starting pipeline. Configuration from ' + cfgfile)

if 'log' in cfg:
    set_log(cfg['log'])

# Preprocess genome, index and allocate disk space for genome structures

igm.Preprocess(cfg)

# Generate random initial configuration

starting_coordinates = cfg.get('starting_coordinates', None) 
if starting_coordinates is not None:
    logger.info('Initial coordinates from: ' + starting_coordinates)
    copyfile(starting_coordinates, cfg['structure_output']) 
else:
    logger.info('Generating random initial coordinates.')
    randomStep = igm.RandomInit(cfg)
    randomStep.run()

    relaxStep = igm.RelaxInit(cfg)
    relaxStep.run()

# optimization iteration
opt_iter = 0
# max unsuccessful optimization iterations before stopping
max_iter = cfg.get('max_iterations', None)

# main optimization loop
while True:
    cfg['runtime']['opt_iter'] = opt_iter

    # setup the needed steps for this optimization iteration
    iter_steps = []
    if 'Hi-C' in cfg['restraints']:

        # prepare the list of sigmas in the runtime status
        if 'sigma_list' not in cfg["runtime"]["Hi-C"]:
            cfg["runtime"]["Hi-C"]['sigma_list'] = cfg["restraints"]["Hi-C"]["sigma_list"][:]
        if "sigma" not in cfg["runtime"]["Hi-C"]:
            cfg["runtime"]["Hi-C"]["sigma"] = cfg["runtime"]["Hi-C"]["sigma_list"].pop(0)

        iter_steps.append(igm.ActivationDistanceStep)

    if 'FISH' in cfg['restraints']:
        iter_steps.append(igm.FishAssignmentStep)
    if 'sprite' in cfg['restraints']:
        iter_steps.append(igm.SpriteAssignmentStep)
    if 'DamID' in cfg['restraints']:
        iter_steps.append(igm.DamidActivationDistanceStep)

    # always run a modeling step
    iter_steps.append(igm.ModelingStep)

    # run the required steps
    for StepClass in iter_steps:
        step = StepClass(cfg)
        step.run()

    # check the violations
    if cfg['runtime']['violation_score'] < 1e-3:
        # no violations, go to next step or finish
        opt_iter = 0
        if 'Hi-C' in cfg['restraints'] and len( cfg["runtime"]["Hi-C"]["sigma_list"] ) != 0:
            # we are done with this sigma but still have more to go
            del cfg["runtime"]["Hi-C"]["sigma"]
        else:
            # no violations, no more work to do
            logger.info('Pipeline completed')
            break
    else:
        # if there are violations, try to optimize again
        opt_iter += 1
        if max_iter is not None:
            if opt_iter >= max_iter:
                logger.critical('Maximum number of iterations reached (%d)' % max_iter)
                break
        logger.info('iteration # %d' % opt_iter)


