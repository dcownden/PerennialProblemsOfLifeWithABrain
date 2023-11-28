
fig, ax = plt.subplots(figsize=(10, 4))

softmax_params = {
    'selection_type': 'softmax',
    'color': 'green',
    'label': 'Softmax',
    'softmax_temp': 7,
}
truncation_params = {
    'selection_type': 'deterministic truncation',
    'color': 'red',
    'label': 'Truncation',
    'truncation_threshold': .9,
}

plot_selection_results(ax, **softmax_params, **common_params)
plot_selection_results(ax, **truncation_params, **common_params)

ax.set_xlabel('Generation')

ax.set_xlabel('Generation')
ax.set_ylabel('Population Score\nMean+IQR')
ax.set_title('Comparison of Selection Types over Generations')
ax.legend()

remove_ip_clutter(fig)
plt.show()