import glob, os

outputfile = 'mooc1_textretrieval_alltranscripts.txt'
# outputfile = 'mooc2_textanalysis_alltranscripts.txt'
# outputfile = 'mooc3_pattern_discovery_alltranscripts.txt'
# outputfile = 'mooc4_cluster_analysis_alltranscripts.txt'

inputdir = "./mooc1-textretrieval-txt/"
# inputdir = "./mooc2-textanalysis-txt/"
# inputdir = "./mooc3-pattern-discovery-txt/"
# inputdir = "./mooc4-cluster-analysis-txt/"

alltranscripts = open(outputfile, 'w')

for file in sorted(os.listdir(inputdir)):
    if file.endswith(".txt"):
    	current = open(inputdir+file, 'r')
    	# print "Current File Being Processed is: " + file
    	alltranscripts.write(current.read().replace('\n', ' '))
    	current.close()
    	alltranscripts.write('\n')

alltranscripts.close()
