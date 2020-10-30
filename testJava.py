from subprocess import *

def jarWrapper(*args):
    process = Popen(['java', '-jar']+list(args), stdout=PIPE, stderr=PIPE, cwd='./Java/Resources')
    ret = []
    while process.poll() is None:
        line = process.stdout.readline().decode("utf-8")
        if line != '' and line.endswith('\n'):
            ret.append(line[:-1])
    stdout, stderr = process.communicate()
    ret.extend(stdout.decode("utf-8").split("\n"))
    if stderr != '':
        ret.extend(stderr.decode("utf-8").split("\n"))
#    ret.remove('')
    ret = list(filter(lambda a: a != "", ret))
    return ret

args = ['erisk-dummy-participant-0.1-jar-with-dependencies.jar', '1'] # Any number of args to be passed to the jar file

print("Program Started...")
result = jarWrapper(*args)
print("Program Ended...")
print(result)