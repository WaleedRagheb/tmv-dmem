from dbm import error
from subprocess import *
from eRisk_T2_runs_dummy import *

def jarWrapper(*args):
    process = Popen(['java', '-jar']+list(args), stdout=PIPE, stderr=PIPE, cwd='./Java/Resources')
    ret = []
    errors = []
    while process.poll() is None:
        line = process.stdout.readline().decode("utf-8")
        if line != '' and line.endswith('\n'):
            ret.append(line[:-1])
    stdout, stderr = process.communicate()
    ret.extend(stdout.decode("utf-8").split("\n"))
    if stderr != '':
        errors.extend(stderr.decode("utf-8").split("\n"))
#    ret.remove('')
    ret = list(filter(lambda a: a != "", ret))
    errors = list(filter(lambda a: a != "", errors))
    return ret, errors

#args = ['erisk-dummy-participant-0.1-jar-with-dependencies.jar', '1'] # Any number of args to be passed to the jar file
#
#print("Program Started...")
#result = jarWrapper(*args)
#print("Program Ended...")
#print(result)

wrDirs = 'Java/Data/dummy/writings'
decDirs = 'Java/Data/dummy/Decisions'

seqNum = -1
args = ['erisk-dummy-participant-0.1-jar-with-dependencies.jar']
while(True):
    ######### Get writings ###############
    getWrArgs = args + ['1']
    result, errors = jarWrapper(*getWrArgs)
    if len(errors) > 0:
        print(error)
        #break

    currentSeq = int(result[len(result)-1])

    ########## Stopping condition ###########
    if(currentSeq == seqNum):
        print('Stopping condition satisfied ...' + str(currentSeq))
        break;
    else:
        print('Processing sequence number ...' + str(currentSeq))
        seqNum = currentSeq

    if((currentSeq > 0) & ((currentSeq % 10) == 0)):
        print('Run the Model to get new decision...... at sequence: [' + str(currentSeq)+ ']')
        eRisk_dummy_runs(wrDirs, decDirs)

    ########## Submit Decisions ############
    decArgs = args + ['4']
    result, errors = jarWrapper(*decArgs)

    if len(errors) > 0:
        print(error)
        break




