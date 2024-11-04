import plotly.graph_objs as go

def HTML3DDrawer(low_dimensional_data, color_index='g', save_path='interactive_3d_plot.html'):
    """_summary_

    Args:
        low_dimensional_data (_type_): the low dimensional input data, three dim restrict
        color_index (_type_): each data point's color, default green
    """

    fig = go.Figure(data=[go.Scatter3d(
        x=low_dimensional_data[:, 0],
        y=low_dimensional_data[:, 1],
        z=low_dimensional_data[:, 2],
        mode='markers',
        marker=dict(
            size=1,
            color=color_index,
            opacity=0.8
        )
    )])

    fig.update_layout(
        scene=dict(
            xaxis_title='Dim1',
            yaxis_title='Dim2',
            zaxis_title='Dim3'
        ),
        title='3D Scatter Plot'
    )

    fig.write_html(save_path)