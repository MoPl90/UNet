import numpy as np
import os as os

import matplotlib.pyplot as plt
import seaborn as sns
from itertools import compress


def print_stats_tables_simple(df, eval_dir, measures=['dice', 'precision', 'recall', 'euclidean_distance'],counts=['truePositiv','trueNegativ','falsePositiv','falseNegativ'],volumes=['lesionLabelVolume','lesionPredictVolume','lesionPredictMinusLabelVolume']):
    out_file = os.path.join(eval_dir, 'stats_tables.txt')

    with open(out_file, "w") as text_file:

        text_file.write('\n\n-------------------------------------------------------------------------------------\n')
        text_file.write('Statistics (mean (standard deviation)):\n')
        text_file.write('-------------------------------------------------------------------------------------\n\n')
        # prints mean (+- std) values for Dice and Jaccard, all structures, averaged over both phases.

        # header_string = ' & '
        header_string = ' '
        line_string = 'METHOD '

        for measure in measures:
            print(measure)
            # header_string += ' ; {}'.format(measure)
            #         eval(df.measure)
            # line_string += ' & {:.3f}\,({:.3f}) '.format(np.mean(df[measure]), np.std(df[measure]))
            line_string = '{}:\t {:.3f} ({:.3f})\n'.format(measure, np.mean(df[measure]), np.std(df[measure]))
            text_file.write(line_string)

        text_file.write('\n\n-------------------------------------------------------------------------------------\n')
        text_file.write('counts: \n')
        text_file.write('-------------------------------------------------------------------------------------\n\n')

        for count in counts:
            print(count)
            line_string = '{}:\t {} \n'.format(count, np.sum(df[count]))
            text_file.write(line_string)

        text_file.write('\n\n-------------------------------------------------------------------------------------\n')
        text_file.write('volumes (mean (standard deviation)): \n')
        text_file.write('-------------------------------------------------------------------------------------\n\n')

        for volume in volumes:
            print(volume)
            line_string = '{}:\t\t {} ({}) \n'.format(volume, np.mean(df[volume]), np.std(df[volume]))
            text_file.write(line_string)


        truePositivList = list(compress(df.to_dict()['caseNumber'].values(), df.to_dict()['truePositiv'].values()))
        trueNegativList = list(compress(df.to_dict()['caseNumber'].values(), df.to_dict()['trueNegativ'].values()))
        falsePositivList = list(compress(df.to_dict()['caseNumber'].values(), df.to_dict()['falsePositiv'].values()))
        falseNegativList = list(compress(df.to_dict()['caseNumber'].values(), df.to_dict()['falseNegativ'].values()))


        text_file.write('\n\n-------------------------------------------------------------------------------------\n')
        text_file.write('true positiv cases: \n')
        text_file.write('-------------------------------------------------------------------------------------\n\n')

        for caseNb in truePositivList:
            print(caseNb)
            line_string = '{} \n'.format(caseNb)
            text_file.write(line_string)



        text_file.write('\n\n-------------------------------------------------------------------------------------\n')
        text_file.write('true negativ cases: \n')
        text_file.write('-------------------------------------------------------------------------------------\n\n')

        for caseNb in trueNegativList:
            print(caseNb)
            line_string = '{} \n'.format(caseNb)
            text_file.write(line_string)



        text_file.write('\n\n-------------------------------------------------------------------------------------\n')
        text_file.write('false positiv cases: \n')
        text_file.write('-------------------------------------------------------------------------------------\n\n')

        for caseNb in falsePositivList:
            print(caseNb)
            line_string = '{} \n'.format(caseNb)
            text_file.write(line_string)



        text_file.write('\n\n-------------------------------------------------------------------------------------\n')
        text_file.write('false negativ cases: \n')
        text_file.write('-------------------------------------------------------------------------------------\n\n')

        for caseNb in falseNegativList:
            print(caseNb)
            line_string = '{} \n'.format(caseNb)
            text_file.write(line_string)



        text_file.write('\n\n-------------------------------------------------------------------------------------\n')
        text_file.write('dice ordered from worst to best : \n')
        text_file.write('-------------------------------------------------------------------------------------\n\n')


        caseNumberList  = list(df.to_dict()['caseNumber'].values())
        diceList  = list(df.to_dict()['dice'].values())

        caseNumberSortedAfterDiceList = [x for _, x in sorted(zip(diceList, caseNumberList))]
        diceSortedList = [x for x in sorted(diceList)]

        for i, caseNb in enumerate(caseNumberSortedAfterDiceList):
            print(caseNb)
            line_string = '{} dice coefficient: {} \n'.format(caseNb,diceSortedList[i])
            text_file.write(line_string)




def print_stats_tables_detailed(df, eval_dir, measures=['dice', 'precision', 'recall'],counts=['truePositiv','trueNegativ','falsePositiv','falseNegativ'],volumes=['lesionLabelVolume','lesionPredictVolume','lesionPredictMinusLabelVolume']):
    out_file = os.path.join(eval_dir, 'stats_tables_detailed.txt')

    with open(out_file, "w") as text_file:

        caseNumberList = list(df.to_dict()['caseNumber'].values())
        diceList = list(df.to_dict()['dice'].values())
        volumeList = list(df.to_dict()['lesionLabelVolume'].values())

        caseNumberSortedAfterDiceList = [x for _, x in sorted(zip(diceList, caseNumberList))]
        volumeSortedListAfterDiceList = [x for _, x in sorted(zip(diceList, volumeList))]
        diceSortedList = [x for x in sorted(diceList)]

        print(diceSortedList)

        for i, caseNb in enumerate(caseNumberSortedAfterDiceList):
            print(caseNb)
            line_string = '{}, {}, {} \n'.format(caseNb, diceSortedList[i],volumeSortedListAfterDiceList[i])
            # line_string = '{}, {} \n'.format(caseNb, diceSortedList[i])
            text_file.write(line_string)








def print_stats_tables(df, eval_dir, structures=['stroke', 'background'], measures=['dice', 'precision', 'recall']):

    """
    Report geometric measures in latex tables to be used in the paper.
    Prints mean (+- std) values for Dice and ASSD for all structures.
    :param df:
    :param eval_dir:
    :return:
    """

    out_file = os.path.join(eval_dir, 'stats_tables.txt')

    with open(out_file, "w") as text_file:

        text_file.write('\n\n-------------------------------------------------------------------------------------\n')
        text_file.write('Statistics:\n')
        text_file.write('-------------------------------------------------------------------------------------\n\n')
        # prints mean (+- std) values for Dice and Jaccard, all structures, averaged over both phases.

        # header_string = ' & '
        header_string = ' '
        line_string = 'METHOD '


        for s_idx, struc_name in enumerate(structures):
            for measure in measures:

                # header_string += ' & {} ({}) '.format(measure, struc_name)
                header_string += ' ; {} ({}) '.format(measure, struc_name)

                dat = df.loc[df['struc'] == struc_name]


                if measure == 'dice':
                    line_string += ' & {:.3f}\,({:.3f}) '.format(np.mean(dat[measure]), np.std(dat[measure]))
                else:
                    line_string += ' & {:.2f}\,({:.2f}) '.format(np.mean(dat[measure]), np.std(dat[measure]))

            if s_idx < 2:
                header_string += ' & '
                line_string += ' & '

        header_string += ' \\\\ \n'
        line_string += ' \\\\ \n'

        text_file.write(header_string)
        text_file.write(line_string)




# def print_latex_tables(df, eval_dir, structures=['endo', 'myo', 'scar', 'rv'], measures=['dice', 'precision', 'recall']):
# def print_latex_tables(df, eval_dir, structures=['stroke', 'background'], measures=['dice', 'precision', 'recall']):
#
#     """
#     Report geometric measures in latex tables to be used in the paper.
#     Prints mean (+- std) values for Dice and ASSD for all structures.
#     :param df:
#     :param eval_dir:
#     :return:
#     """
#
#     out_file = os.path.join(eval_dir, 'latex_tables.txt')
#
#     with open(out_file, "w") as text_file:
#
#         text_file.write('\n\n-------------------------------------------------------------------------------------\n')
#         text_file.write('Statistics:\n')
#         text_file.write('-------------------------------------------------------------------------------------\n\n')
#         # prints mean (+- std) values for Dice and Jaccard, all structures, averaged over both phases.
#
#         header_string = ' & '
#         line_string = 'METHOD '
#
#
#         for s_idx, struc_name in enumerate(structures):
#             for measure in measures:
#
#                 header_string += ' & {} ({}) '.format(measure, struc_name)
#
#                 dat = df.loc[df['struc'] == struc_name]
#
#                 if measure == 'dice':
#                     line_string += ' & {:.3f}\,({:.3f}) '.format(np.mean(dat[measure]), np.std(dat[measure]))
#                 else:
#                     line_string += ' & {:.2f}\,({:.2f}) '.format(np.mean(dat[measure]), np.std(dat[measure]))
#
#             if s_idx < 2:
#                 header_string += ' & '
#                 line_string += ' & '
#
#         header_string += ' \\\\ \n'
#         line_string += ' \\\\ \n'
#
#         text_file.write(header_string)
#         text_file.write(line_string)


def boxplot_metrics(df, eval_dir, measures=['dice', 'precision', 'recall']):
    """
    Create summary boxplots of all geometric measures.
    :param df:
    :param eval_dir:
    :return:
    """

    no_meas = len(measures)

    #######################################################
    # catplots_file = os.path.join(eval_dir, 'catplots.png')
    # fig, axes = plt.subplots(no_meas, 1)
    # fig.set_figheight(14)
    # fig.set_figwidth(7)
    # for ii in range(no_meas):
    #     sns.catplot(x='struc', y=measures[ii], data=df, palette='PRGn', ax=axes[ii])
    #     plt.close(2)
    # plt.savefig(catplots_file)
    # plt.close()

    #######################################################

    boxplots_file = os.path.join(eval_dir, 'boxplots.png')
    fig, axes = plt.subplots(no_meas, 1)
    fig.set_figheight(14)
    fig.set_figwidth(7)
    for ii in range(no_meas):
        sns.boxplot(x='struc', y=measures[ii], data=df, palette='PRGn', ax=axes[ii])
    # sns.boxplot(x='struc', y='dice', data=df, palette="PRGn", ax=axes[0])
    # sns.boxplot(x='struc', y='precision', data=df, palette="PRGn", ax=axes[1])
    # sns.boxplot(x='struc', y='recall', data=df, palette="PRGn", ax=axes[2])
    plt.savefig(boxplots_file)
    plt.close()

    #######################################################

    return 0


# def print_stats(df, eval_dir, structures=['endo', 'myo', 'scar', 'rv'], measures=['dice', 'precision', 'recall']):
def print_stats(df, eval_dir, structures=['stroke', 'background'], measures=['dice', 'precision', 'recall']):
    out_file = os.path.join(eval_dir, 'summary_report.txt')

    with open(out_file, "w") as text_file:

        text_file.write('\n\n-------------------------------------------------------------------------------------\n')
        text_file.write('Summary of geometric evaluation measures. \n')
        text_file.write('-------------------------------------------------------------------------------------\n\n')

        for struc_name in structures:

            text_file.write(struc_name)
            text_file.write('\n')


            for measure_name in measures:

                dat = df.loc[df['struc'] == struc_name]
                text_file.write('       {} -- mean (std): {:.3f} ({:.3f}) \n'.format(measure_name,
                                                                         np.mean(dat[measure_name].dropna()), np.std(dat[measure_name].dropna())))

                ind_med = np.argsort(dat[measure_name]).iloc[len(dat[measure_name])//2]
                text_file.write('             median {}: {:.3f} ({})\n'.format(measure_name,
                                                                dat[measure_name].dropna().iloc[ind_med], dat['filename'].dropna().iloc[ind_med]))

                ind_worst = np.argsort(dat[measure_name]).iloc[0]
                text_file.write('             worst {}: {:.3f} ({})\n'.format(measure_name,
                                                                dat[measure_name].dropna().iloc[ind_worst], dat['filename'].dropna().iloc[ind_worst]))

                ind_best = np.argsort(dat[measure_name]).iloc[-1]
                text_file.write('             best {}: {:.3f} ({})\n'.format(measure_name,
                                                                dat[measure_name].dropna().iloc[ind_best], dat['filename'].dropna().iloc[ind_best]))

