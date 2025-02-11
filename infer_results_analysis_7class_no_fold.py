import pandas as pd
from bokeh.plotting import figure as Figure
from bokeh.layouts import gridplot
from bokeh.io import show, save
from pathlib import Path
import lightkurve as lk
from lightkurve.periodogram import LombScarglePeriodogram


def main():
    def infer_results_classdf(file='infer_result_99_whole.csv'):
        """
        :param file: a csv file of inferred results of NN
        :return: a dictionary of dataframes for each class and their probabilities, given the probabilities are greater than 0.5.
        """
        column_dataframes = {}
        infer_data = pd.read_csv(file)
        split_data = infer_data['confidence'].str.strip('[]').str.split(expand=True)
        infer_data = pd.concat([infer_data, split_data], axis=1)
        infer_data = infer_data.drop('confidence', axis=1)
        infer_data.rename(
            columns={0: 'ds_p',
                     1: 'rrl_p',
                     2: 'rm_p',
                     3: 'eb_p',
                     4: 'c_p',
                     5: 'n_p'}, inplace=True)
        class_list = list(infer_data.columns[1::])
        for column in class_list:
            infer_data[column] = infer_data[column].astype(float)
            filtered_df = infer_data[infer_data[column] > 0.5][['path', column]]
            column_name = f"{column}"
            column_dataframes[column_name] = filtered_df
        return column_dataframes


    def visualize_infer_results(column_dataframes):
        """
        :param column_dataframes: The returned result of the infer_results_classdf() function which is a dictionary of dataframes of each classes
        :return: the grid of images of the star candidates of each class predicted by the model
        """
        figures = []
        for star_class, class_df in column_dataframes.items():
            class_df = class_df.sort_values(by=class_df.columns[1], ascending=False)
            class_df['path'] = class_df['path'].astype(str)
            paths = class_df['path'].tolist()
            for path in paths[:50]:
                def plot_folded_light_curve(path):
                    light_curve = lk.read(Path(path))
                    light_curve = light_curve.remove_outliers(sigma=5)
                    fluxes = light_curve.flux.value
                    times = light_curve.time.value
                    #period = find_period_lk(light_curve)
                    #folded_light_curve = light_curve.fold(period=period)
                    light_curve_figure = Figure(x_axis_label='Time', y_axis_label='Flux',
                                                title=str(star_class + '_' + path[-40:]))
                    light_curve_figure.scatter(x=light_curve.time.value, y=light_curve.flux.value)
                    light_curve_figure.line(x=light_curve.time.value, y=light_curve.flux.value,
                                            line_alpha=0.3)
                    return light_curve_figure

                figures.append(plot_folded_light_curve(path))

        grid = gridplot(figures, ncols=5, width=350, height=250)
        return grid

    grid = visualize_infer_results(infer_results_classdf())
    save(grid)


if __name__ == '__main__':
    main()
