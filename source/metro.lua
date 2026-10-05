-- Created by: Arena.ai Agent Mode (AI) - NightCity MTA:SA resource
-- -------------------------------------------------------------------------------------------
-- metro.lua - two independent rideable Night City Metro loops, on the lower tunnel and upper
--             express levels. Each uses the VANILLA GTA:SA consist (model 538 "streak" engine
--             + three 570 "streakc" carriages). Both call at MARKET / UNION / DOCKS with dwells.
--             Press E next to a train to board / exit; /ncmetro [1|2] selects a level.
--             Uses the helpers client.lua publishes in the global table NC.
-- -------------------------------------------------------------------------------------------
local ACC = 1.2            -- m/s^2 while motoring
local BRK = 1.4            -- m/s^2 while braking
local CAR_GAP = 23.5       -- centre to centre between cars
local CARS = { 538, 570, 570, 570 }   -- vanilla GTA:SA train models
local DECK_Z = 0.62        -- running surface above the deck datum (G.z)
local RIDE_Z = 1.05        -- carriage floor: the rider stands INSIDE a car, not on its roof

local function newState()
    return { on = false, s = 0.0, v = 0.0, targetS = 0.0, dwell = 0.0,
             cars = nil, riding = false, rideCar = 1, last = 0.0, lift = 1.15, frameFn = nil }
end

local M, M2 = newState(), newState()
local G, G2 = {}, {}
local LINES = {
    { state = M, geom = G, cfg = NC_METRO, label = "Lower" },
    { state = M2, geom = G2, cfg = NC_METRO2, label = "Upper" },
}
local rideBound = false

local function buildGeom(t, g)
    if not t then return false end
    g.x0, g.x1, g.y, g.ys, g.r = t.x0, t.x1, t.y, t.ys, t.r
    g.z = t.z
    g.ym = (t.y + t.ys) / 2
    g.l1 = t.x1 - t.x0
    g.la = math.pi * t.r
    g.l2 = g.l1
    g.lb = g.la
    g.total = 2 * (g.l1 + g.la)
    g.stops = {}
    for i = 1, #t.stops do g.stops[i] = t.stops[i] - t.x0 end
    return true
end

-- Position + travel direction at loop distance s (forward is east along the north straight).
local function pathAt(g, s)
    s = s % g.total
    if s < g.l1 then
        return g.x0 + s, g.y, 1.0, 0.0
    elseif s < g.l1 + g.la then
        local phi = math.pi / 2 - (s - g.l1) / g.r
        return g.x1 + g.r * math.cos(phi), g.ym + g.r * math.sin(phi), math.sin(phi), -math.cos(phi)
    elseif s < g.l1 + g.la + g.l2 then
        return g.x1 - (s - g.l1 - g.la), g.ys, -1.0, 0.0
    else
        local phi = -math.pi / 2 - (s - g.l1 - g.la - g.l2) / g.r
        return g.x0 + g.r * math.cos(phi), g.ym + g.r * math.sin(phi), math.sin(phi), -math.cos(phi)
    end
end

local function say(msg) if NC and NC.say then NC.say(msg) end end

local function nextStop(g, cur)
    for lap = 0, 4 do
        for i = 1, #g.stops do
            local s = g.stops[i] + lap * g.total
            if s > cur + 0.5 then return s end
        end
    end
    return cur + g.total
end

local function placeCars(state, g)
    if not state.cars then return end
    for i, veh in ipairs(state.cars) do
        if isElement(veh) then
            local x, y, dx, dy = pathAt(g, state.s - (i - 1) * CAR_GAP)
            setElementPosition(veh, NC.toWorld(x, y, g.z + DECK_Z + state.lift))
            setElementRotation(veh, 0, 0, math.deg(math.atan2(-dx, dy)))
        end
    end
end

local function measureLift(state, veh)
    local ok, _, _, minZ = pcall(getElementBoundingBox, veh)
    if ok and type(minZ) == "number" and minZ < -0.1 and minZ > -6.0 then
        state.lift = -minZ
    else
        state.lift = 1.15
    end
end

local function releaseRider(state)
    if state.riding then
        state.riding = false
        setElementFrozen(localPlayer, false)
        if state.cars then
            for _, v in ipairs(state.cars) do
                if isElement(v) then setElementCollisionsEnabled(v, true) end
            end
        end
    end
end

local function makeFrame(state, g, cfg)
    return function()
        if not (state.on and state.cars) then return end
        local now = getTickCount()
        local dt = (now - state.last) / 1000
        state.last = now
        if dt > 0.1 then dt = 0.1 end
        if dt <= 0 then return end

        local vmax = cfg.vx or 16.0
        if state.dwell > 0 then
            state.dwell = state.dwell - dt
        else
            local d = state.targetS - state.s
            local ad = math.abs(d)
            local vpeak = math.min(vmax, math.sqrt(math.max(0.0, 2 * ad * ACC * BRK / (ACC + BRK))))
            if state.v * state.v / (2 * BRK) >= ad - 0.25 then
                state.v = math.max(0.0, state.v - BRK * dt)
            else
                state.v = math.min(vpeak, state.v + ACC * dt)
            end
            if state.v * dt >= ad or (state.v <= 0.0 and ad < 1.0) then
                state.s = state.targetS
                state.v = 0.0
                state.dwell = cfg.dw or 6.0
                state.targetS = nextStop(g, state.s)
            else
                state.s = state.s + state.v * dt
            end
        end

        placeCars(state, g)
        if state.riding and state.cars[state.rideCar] and isElement(state.cars[state.rideCar]) then
            local x, y = pathAt(g, state.s - (state.rideCar - 1) * CAR_GAP)
            setElementPosition(localPlayer, NC.toWorld(x, y, g.z + RIDE_Z))
        end
    end
end

local function leaveLine(line)
    local state, g = line.state, line.geom
    releaseRider(state)
    local x, y = pathAt(g, state.s - (state.rideCar - 1) * CAR_GAP)
    local outY, outZ = g.y + 2.6, 0.06
    if state.dwell > 0 then outY, outZ = g.y + 8.5, 1.02 end
    setElementPosition(localPlayer, NC.toWorld(x, outY, g.z + outZ))
    say("left the " .. line.label:lower() .. " metro level.")
end

local function toggleRide()
    for _, line in ipairs(LINES) do
        if line.state.riding then leaveLine(line) return end
    end
    local px, py, pz = NC.toCity(getElementPosition(localPlayer))
    local best, bestScore, bestCar
    for _, line in ipairs(LINES) do
        local state, g = line.state, line.geom
        if state.on and state.cars then
            for i = 1, #state.cars do
                local x, y = pathAt(g, state.s - (i - 1) * CAR_GAP)
                local dx, dy, dz = px - x, py - y, pz - g.z
                if math.abs(dx) < 12.0 and math.abs(dy) < 8.0 and dz > -2.5 and dz < 5.5 then
                    local score = dx * dx + dy * dy + 0.35 * dz * dz
                    if not bestScore or score < bestScore then
                        best, bestScore, bestCar = line, score, i
                    end
                end
            end
        end
    end
    if not best then return end
    local state, g = best.state, best.geom
    if NC.stopTour then NC.stopTour() end
    state.riding, state.rideCar = true, bestCar
    local x, y = pathAt(g, state.s - (bestCar - 1) * CAR_GAP)
    setElementFrozen(localPlayer, true)
    setElementPosition(localPlayer, NC.toWorld(x, y, g.z + RIDE_Z))
    for _, v in ipairs(state.cars) do
        if isElement(v) then setElementCollisionsEnabled(v, false) end
    end
    say("aboard the " .. best.label:lower() .. " Night City Metro - press E to leave.")
end

local function destroyCars(state)
    if state.cars then
        for _, veh in ipairs(state.cars) do if isElement(veh) then destroyElement(veh) end end
    end
    state.cars = nil
end

local function startLine(line)
    local state, g, cfg = line.state, line.geom, line.cfg
    if state.on or not buildGeom(cfg, g) then return end
    state.cars = {}
    for i, model in ipairs(CARS) do
        local x, y, dx, dy = pathAt(g, 150.0 - (i - 1) * CAR_GAP)
        local wx, wy, wz = NC.toWorld(x, y, g.z + DECK_Z)
        local veh = createVehicle(model, wx, wy, wz, 0, 0, math.deg(math.atan2(-dx, dy)))
        if veh then
            if i == 1 then measureLift(state, veh) end
            setElementFrozen(veh, true)
            pcall(setVehicleDamageProof, veh, true)
            pcall(setVehicleLocked, veh, true)
            pcall(setVehicleEngineState, veh, false)
            setElementCollisionsEnabled(veh, true)
            state.cars[#state.cars + 1] = veh
        end
    end
    if #state.cars == 0 then
        state.cars = nil
        say("metro " .. line.label:lower() .. ": could not create the train.")
        return
    end
    state.on = true
    state.s, state.v, state.dwell = 150.0, 0.0, 0.0
    state.targetS = nextStop(g, state.s)
    state.last = getTickCount()
    placeCars(state, g)
    state.frameFn = makeFrame(state, g, cfg)
    addEventHandler("onClientPreRender", root, state.frameFn)
end

local function start()
    for _, line in ipairs(LINES) do startLine(line) end
    if (M.on or M2.on) and not rideBound then
        bindKey("e", "down", toggleRide)
        rideBound = true
    end
end

local function stopLine(state)
    if not state.on and not state.cars then return end
    if state.frameFn then removeEventHandler("onClientPreRender", root, state.frameFn) end
    state.frameFn = nil
    releaseRider(state)
    state.on = false
    destroyCars(state)
end

local function stop()
    for _, line in ipairs(LINES) do stopLine(line.state) end
    if rideBound then unbindKey("e", "down", toggleRide) rideBound = false end
end

NC.onHide = stop
NC.onShow = start

NC_METRO_RT = M
NC_METRO_UPPER_RT = M2

addCommandHandler("ncmetro", function(_, which)
    if not (NC and NC.isShown()) then return say("show the city first (/ncshow).") end
    local upper = type(which) == "string" and (which:lower() == "2" or which:lower() == "upper" or which:lower() == "high")
    local line = upper and LINES[2] or LINES[1]
    if M.riding or M2.riding then toggleRide() end
    local cfg = line.cfg
    if not cfg then return say("that metro level is not available in this build.") end
    setElementPosition(localPlayer, NC.toWorld(cfg.stops[1], cfg.y + 8.5, cfg.z + 1.02))
    say("Night City Metro " .. (upper and "Upper" or "Lower") .. ": press E near the train to ride. Use /ncmetro 1 or /ncmetro 2 to switch levels.")
end)
