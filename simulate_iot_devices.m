function meta = simulate_iot_devices(N)
    % simple metadata: id, type, reliability (random)
    types = {'sensor','actuator','gateway','camera','thermostat'};
    meta = struct();
    meta.N = N;
    meta.ids = (1:N)';
    meta.type = types(randi(numel(types),N,1));
    meta.reliability = rand(N,1); % 0..1
end
